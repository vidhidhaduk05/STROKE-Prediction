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

function displayResults(probTreated, probControl, features) {
    const cate = probTreated - probControl;
    const benefitScore = computeBenefitScore(cate);
    const absDiff = Math.abs((probTreated - probControl) * 100);

    // Show results panel
    document.getElementById('results').style.display = 'block';

    // Honest coloring: green = meaningfully better arm, red = worse arm,
    // neutral grey when the difference is within the +/-5% benefit threshold.
    const threshold = MODEL.recommendation_threshold;
    const GREEN = '#2f7d32', RED = '#a0352c', NEUTRAL = '#6b6b6b';
    let colTreated, colControl, stateTreated, stateControl;
    if (cate > threshold) {
        colTreated = GREEN; colControl = RED;
        stateTreated = 'card-good'; stateControl = 'card-bad';
    } else if (cate < -threshold) {
        colTreated = RED; colControl = GREEN;
        stateTreated = 'card-bad'; stateControl = 'card-good';
    } else {
        colTreated = NEUTRAL; colControl = NEUTRAL;
        stateTreated = 'card-neutral'; stateControl = 'card-neutral';
    }

    // Gauges
    setGauge('gauge-arc-treated', 'gauge-pct-treated', probTreated, colTreated);
    setGauge('gauge-arc-control', 'gauge-pct-control', probControl, colControl);

    // Outcome score cards (each arm's probability mapped to 1-20)
    document.getElementById('score-treated-val').textContent = probToScore(probTreated) + ' /20';
    document.getElementById('score-control-val').textContent = probToScore(probControl) + ' /20';
    document.getElementById('card-treated').className = 'score-card ' + stateTreated;
    document.getElementById('card-control').className = 'score-card ' + stateControl;

    // Advantage banner — driven by the same +/-5% benefit threshold
    const banner = document.getElementById('advantage-banner');
    if (cate > threshold) {
        banner.textContent = 'Aspirin favored: +' + (cate * 100).toFixed(1) + '% favorable-outcome probability';
        banner.className = 'advantage-banner adv-positive';
    } else if (cate < -threshold) {
        banner.textContent = 'No aspirin favored: ' + (cate * 100).toFixed(1) + '% (aspirin worse)';
        banner.className = 'advantage-banner adv-negative';
    } else {
        banner.textContent = 'No meaningful difference between arms (within \u00B15%)';
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

    // Explainability (feature / interaction contributions)
    if (features) displayExplainability(features, probControl, cate);

    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---- Explainability ----
const FEATURE_LABELS = {
    age_gt80: 'Age over 80',
    male: 'Male sex',
    afib: 'Atrial fibrillation',
    sbp_gt180: 'SBP over 180 mmHg',
    impaired_conscious: 'Impaired consciousness',
    delay_gt6h: 'Delay over 6 hours',
    tacs: 'Total anterior circulation stroke (TACS)',
    deficit_ge3: '3 or more neurological deficits',
    prior_aspirin: 'Prior aspirin use',
    infarct_visible: 'Visible infarct on scan',
    heparin_allocated: 'Heparin allocated',
};

function renderDriverBars(containerId, contribs) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    if (!contribs.length) {
        container.innerHTML = '<p class="explain-desc">No contributing factors for this patient.</p>';
        return;
    }
    const maxAbs = Math.max.apply(null, contribs.map(c => Math.abs(c.value)).concat([1e-6]));
    contribs.forEach(c => {
        const row = document.createElement('div');
        row.className = 'driver-row';

        const label = document.createElement('div');
        label.className = 'driver-label';
        label.textContent = c.label;

        const track = document.createElement('div');
        track.className = 'driver-track';
        const bar = document.createElement('div');
        const pct = (Math.abs(c.value) / maxAbs) * 50; // half-track = 50%
        bar.className = 'driver-bar ' + (c.value >= 0 ? 'pos' : 'neg');
        if (c.value >= 0) {
            bar.style.left = '50%';
            bar.style.width = pct + '%';
        } else {
            bar.style.width = pct + '%';
            bar.style.left = (50 - pct) + '%';
        }
        track.appendChild(bar);

        const val = document.createElement('div');
        val.className = 'driver-val ' + (c.value >= 0 ? 'pos' : 'neg');
        val.textContent = (c.value >= 0 ? '+' : '') + c.value.toFixed(3);

        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(val);
        container.appendChild(row);
    });
}

function displayExplainability(features, probControl, cate) {
    const names = MODEL.feature_names;
    const coefs = MODEL.coefficients;
    const idx = {};
    names.forEach((n, i) => { idx[n] = i; });

    // --- Treatment-effect decomposition: base effect + active interactions ---
    const treatContribs = [{
        label: 'Aspirin base effect (all patients)',
        value: coefs[idx['treatment']],
    }];
    names.forEach((n, i) => {
        if (n.startsWith('treat_x_')) {
            const base = n.replace('treat_x_', '');
            if (features[base] === 1 && Math.abs(coefs[i]) > 1e-9) {
                treatContribs.push({ label: 'Aspirin \u00D7 ' + FEATURE_LABELS[base], value: coefs[i] });
            }
        }
    });
    renderDriverBars('treat-drivers', treatContribs);

    const netLogit = treatContribs.reduce((s, c) => s + c.value, 0);
    const cateSign = cate >= 0 ? '+' : '';
    document.getElementById('treat-net').innerHTML =
        'Net treatment effect: <strong>' + (netLogit >= 0 ? '+' : '') + netLogit.toFixed(3) +
        ' log-odds &nbsp;&rarr;&nbsp; ' + cateSign + (cate * 100).toFixed(2) + '% probability</strong>';

    // --- Plain-language summary of the decision ---
    const threshold = MODEL.recommendation_threshold;
    const sorted = treatContribs.slice(1).sort((a, b) => a.value - b.value);
    const biggestNeg = sorted.length ? sorted[0] : null;
    const base = coefs[idx['treatment']];
    let plain;
    if (cate > threshold) {
        plain = 'For this patient, aspirin\u2019s effect is net positive (+' + (cate * 100).toFixed(1) +
            '%), so the model favours giving aspirin.';
    } else if (cate < -threshold) {
        plain = 'For this patient, aspirin\u2019s small base benefit (+' + base.toFixed(3) + ') is outweighed' +
            (biggestNeg ? ' by the negative interaction with \u201C' + biggestNeg.label.replace('Aspirin \u00D7 ', '') +
            '\u201D (' + biggestNeg.value.toFixed(3) + ')' : '') +
            ', giving a net effect of ' + (cate * 100).toFixed(1) + '%. The model favours withholding aspirin.';
    } else {
        plain = 'Aspirin\u2019s base benefit (+' + base.toFixed(3) + ' log-odds) is ' +
            (biggestNeg && biggestNeg.value < 0
                ? 'offset by the negative interaction with \u201C' + biggestNeg.label.replace('Aspirin \u00D7 ', '') +
                  '\u201D (' + biggestNeg.value.toFixed(3) + '), leaving a net effect of '
                : 'not reinforced by any positive interaction, leaving a net effect of ') +
            (cate * 100).toFixed(1) + '%. Because this lies within the \u00B1' + (threshold * 100).toFixed(0) +
            '% threshold, there is no clear benefit either way \u2014 the decision rests on clinical judgement, ' +
            'bleeding risk, and patient preference rather than this score.';
    }
    document.getElementById('treat-plain').textContent = plain;

    // --- Outcome (baseline) drivers: intercept + active base features ---
    const outContribs = [{ label: 'Baseline (intercept)', value: MODEL.intercept }];
    Object.keys(FEATURE_LABELS).forEach(base => {
        if (features[base] === 1 && idx[base] !== undefined && Math.abs(coefs[idx[base]]) > 1e-9) {
            outContribs.push({ label: FEATURE_LABELS[base], value: coefs[idx[base]] });
        }
    });
    outContribs.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    renderDriverBars('outcome-drivers', outContribs.slice(0, 7));
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

    displayResults(probTreated, probControl, features);
}
