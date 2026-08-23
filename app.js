/**
 * Stroke Treatment Benefit Predictor — Client-side prediction engine.
 *
 * Loads the exported Elastic Net model (model.json), collects patient
 * inputs, binarizes features, builds the feature vector with treatment
 * interaction terms, and computes counterfactual probabilities.
 *
 * Logistic regression: P = sigmoid(intercept + sum(coef_i * x_i))
 */

let MODEL = null;

// Load model on page ready
document.addEventListener('DOMContentLoaded', () => {
    fetch('model.json')
        .then(r => r.json())
        .then(data => { MODEL = data; })
        .catch(err => {
            console.error('Failed to load model:', err);
            alert('Error: could not load model.json. Make sure you are serving via http (not file://).');
        });
});

function sigmoid(x) {
    if (x >= 0) return 1 / (1 + Math.exp(-x));
    const e = Math.exp(x);
    return e / (1 + e);
}

// ---- Sample patient data ----
const SAMPLE_PATIENT = {
    age: 72,
    sex: 'M',
    sbp: 165,
    delay: 5,
    consc: 'F',
    stype: 'PACS',
    afib: 'N',
    prior_asp: 'N',
    infarct: 'N',
    heparin: 'N',
    deficits: [true, true, false, true, false, false, false, false], // Face, Arm, Dysphasia
};

function loadSample() {
    document.getElementById('age').value = SAMPLE_PATIENT.age;
    document.getElementById('sex').value = SAMPLE_PATIENT.sex;
    document.getElementById('sbp').value = SAMPLE_PATIENT.sbp;
    document.getElementById('delay').value = SAMPLE_PATIENT.delay;
    document.getElementById('consc').value = SAMPLE_PATIENT.consc;
    document.getElementById('stype').value = SAMPLE_PATIENT.stype;
    document.getElementById('afib').value = SAMPLE_PATIENT.afib;
    document.getElementById('prior_asp').value = SAMPLE_PATIENT.prior_asp;
    document.getElementById('infarct').value = SAMPLE_PATIENT.infarct;
    document.getElementById('heparin').value = SAMPLE_PATIENT.heparin;

    const defIds = ['def_face', 'def_arm', 'def_leg', 'def_dysph',
                    'def_hemian', 'def_vspatial', 'def_brainstem', 'def_other'];
    defIds.forEach((id, i) => {
        document.getElementById(id).checked = SAMPLE_PATIENT.deficits[i];
    });

    document.getElementById('results').style.display = 'none';
}

// ---- Input collection ----
function collectInputs() {
    const defIds = ['def_face', 'def_arm', 'def_leg', 'def_dysph',
                    'def_hemian', 'def_vspatial', 'def_brainstem', 'def_other'];
    const deficits = defIds.map(id => document.getElementById(id).checked);

    return {
        age: parseFloat(document.getElementById('age').value),
        sex: document.getElementById('sex').value,
        sbp: parseFloat(document.getElementById('sbp').value),
        delay: parseFloat(document.getElementById('delay').value),
        consc: document.getElementById('consc').value,
        stype: document.getElementById('stype').value,
        deficits: deficits,
        afib: document.getElementById('afib').value,
        prior_asp: document.getElementById('prior_asp').value,
        infarct: document.getElementById('infarct').value,
        heparin: document.getElementById('heparin').value,
    };
}

// ---- Feature binarization ----
function binarizeFeatures(input) {
    const t = MODEL.thresholds;
    const deficitCount = input.deficits.filter(d => d).length;

    return {
        age_gt80: input.age > t.age ? 1 : 0,
        male: input.sex === 'M' ? 1 : 0,
        afib: input.afib === 'Y' ? 1 : 0,
        sbp_gt180: input.sbp > t.sbp ? 1 : 0,
        impaired_conscious: (input.consc === 'D' || input.consc === 'U') ? 1 : 0,
        delay_gt6h: input.delay > t.delay ? 1 : 0,
        tacs: input.stype === 'TACS' ? 1 : 0,
        deficit_ge3: deficitCount >= t.deficit_count ? 1 : 0,
        prior_aspirin: input.prior_asp === 'Y' ? 1 : 0,
        infarct_visible: input.infarct === 'Y' ? 1 : 0,
        heparin_allocated: input.heparin === 'Y' ? 1 : 0,
    };
}

// ---- Feature vector construction ----
function buildFeatureVector(features, treatmentValue) {
    const featureNames = MODEL.feature_names;
    const vector = [];

    for (const name of featureNames) {
        if (name === 'treatment') {
            vector.push(treatmentValue);
        } else if (name.startsWith('treat_x_')) {
            const baseName = name.replace('treat_x_', '');
            vector.push(treatmentValue * features[baseName]);
        } else {
            vector.push(features[name]);
        }
    }
    return vector;
}

// ---- Prediction ----
function predictProbability(featureVector) {
    let logit = MODEL.intercept;
    for (let i = 0; i < MODEL.coefficients.length; i++) {
        logit += MODEL.coefficients[i] * featureVector[i];
    }
    return sigmoid(logit);
}

// ---- Treatment Benefit Score ----
function computeBenefitScore(cate) {
    const p1 = MODEL.benefit_score.cate_p1;
    const p99 = MODEL.benefit_score.cate_p99;
    const minScore = MODEL.benefit_score.min_score;
    const maxScore = MODEL.benefit_score.max_score;

    if (p99 === p1) return (minScore + maxScore) / 2;

    let normalized = (cate - p1) / (p99 - p1);
    normalized = Math.max(0, Math.min(1, normalized));
    return minScore + normalized * (maxScore - minScore);
}

// ---- Recommendation ----
function getRecommendation(cate) {
    const threshold = MODEL.recommendation_threshold;
    if (cate > threshold) {
        return {
            text: 'Aspirin recommended',
            detail: 'Treatment benefit: +' + (cate * 100).toFixed(1) + '% above no-aspirin',
            class: 'rec-treat',
        };
    } else if (cate < -threshold) {
        return {
            text: 'No aspirin recommended',
            detail: 'Treatment benefit: ' + (cate * 100).toFixed(1) + '% (aspirin worse)',
            class: 'rec-control',
        };
    } else {
        return {
            text: 'No clear treatment benefit',
            detail: 'Differential benefit within +/-' + (threshold * 100).toFixed(0) + '% threshold',
            class: 'rec-neutral',
        };
    }
}

// ---- Results display ----
const GAUGE_LEN = Math.PI * 80; // semicircle arc length, r = 80

function setGauge(arcId, pctId, prob, color) {
    const arc = document.getElementById(arcId);
    const len = Math.max(0, Math.min(1, prob)) * GAUGE_LEN;
    arc.setAttribute('stroke-dasharray', len.toFixed(1) + ' ' + GAUGE_LEN.toFixed(1));
    arc.setAttribute('stroke', color);
    document.getElementById(pctId).textContent = (prob * 100).toFixed(0) + '%';
}

function probToScore(prob) {
    // Map predicted probability (0-1) to a 1-20 clinical score
    return Math.max(1, Math.min(20, Math.round(1 + prob * 19)));
}

function displayResults(probTreated, probControl) {
    const cate = probTreated - probControl;
    const benefitScore = computeBenefitScore(cate);
    const absDiff = Math.abs((probTreated - probControl) * 100);

    // Show results panel
    document.getElementById('results').style.display = 'block';

    // Gauges
    setGauge('gauge-arc-treated', 'gauge-pct-treated', probTreated, '#2f7d32');
    setGauge('gauge-arc-control', 'gauge-pct-control', probControl, '#a0352c');

    // Score cards (probability mapped to 1-20)
    const scoreTreated = probToScore(probTreated);
    const scoreControl = probToScore(probControl);
    document.getElementById('score-treated-val').textContent = scoreTreated + ' /20';
    document.getElementById('score-control-val').textContent = scoreControl + ' /20';

    // Advantage banner
    const diff = scoreTreated - scoreControl;
    const banner = document.getElementById('advantage-banner');
    if (diff > 0) {
        banner.textContent = 'Aspirin advantage: +' + diff + ' points';
        banner.className = 'advantage-banner adv-positive';
    } else if (diff < 0) {
        banner.textContent = 'No-aspirin advantage: +' + (-diff) + ' points';
        banner.className = 'advantage-banner adv-negative';
    } else {
        banner.textContent = 'No score difference between arms';
        banner.className = 'advantage-banner adv-neutral';
    }

    // CATE marker on zone track (track spans -10% to +10%)
    const trackPos = Math.max(0, Math.min(100, ((cate + 0.10) / 0.20) * 100));
    document.getElementById('cate-marker').style.left = trackPos + '%';

    // Metrics
    const cateSign = cate >= 0 ? '+' : '';
    document.getElementById('cate-val').textContent = cateSign + (cate * 100).toFixed(2) + '%';
    document.getElementById('diff-val').textContent = absDiff.toFixed(2) + '%';
    document.getElementById('score-val').textContent = benefitScore.toFixed(1) + ' / 20';

    // Recommendation
    const rec = getRecommendation(cate);
    const recCard = document.getElementById('rec-card');
    recCard.className = 'tool-card ' + rec.class;
    document.getElementById('rec-text').textContent = rec.text;
    document.getElementById('rec-detail').textContent = rec.detail;

    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---- Main calculation ----
function calculate() {
    if (!MODEL) {
        alert('Model not loaded yet. Please wait a moment and try again.');
        return;
    }

    const input = collectInputs();
    const features = binarizeFeatures(input);

    // Counterfactual: predict under both treatment scenarios
    const vecTreated = buildFeatureVector(features, 1);
    const vecControl = buildFeatureVector(features, 0);

    const probTreated = predictProbability(vecTreated);
    const probControl = predictProbability(vecControl);

    displayResults(probTreated, probControl);
}
