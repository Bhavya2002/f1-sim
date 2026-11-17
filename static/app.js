/**
 * Formula 1 Race Strategy What-If Simulator Frontend Logic.
 */

let raceInfo = null;
let baselineStints = {};
let officialClassification = [];
let selectedDriver = 55; // Default: Carlos Sainz
let currentStints = [];
let activePreset = 'baseline';
let activeStrategyScenario = null; // 'ham_no_stop' | 'norris_2stop' | 'sainz_early' | null

function getStrategyStintsForScenario(scenarioName) {
    const totalLaps = (raceInfo && raceInfo.total_laps) ? raceInfo.total_laps : 62;
    if (scenarioName === 'ham_no_stop') {
        return [
            { stint_number: 1, lap_start: 1, lap_end: 20, compound: 'MEDIUM', start_tyre_age: 0 },
            { stint_number: 2, lap_start: 21, lap_end: totalLaps, compound: 'HARD', start_tyre_age: 0 }
        ];
    } else if (scenarioName === 'norris_2stop') {
        return [
            { stint_number: 1, lap_start: 1, lap_end: 20, compound: 'MEDIUM', start_tyre_age: 0 },
            { stint_number: 2, lap_start: 21, lap_end: 44, compound: 'HARD', start_tyre_age: 0 },
            { stint_number: 3, lap_start: 45, lap_end: totalLaps, compound: 'SOFT', start_tyre_age: 0 }
        ];
    } else if (scenarioName === 'sainz_early') {
        return [
            { stint_number: 1, lap_start: 1, lap_end: 15, compound: 'MEDIUM', start_tyre_age: 0 },
            { stint_number: 2, lap_start: 16, lap_end: totalLaps, compound: 'HARD', start_tyre_age: 0 }
        ];
    }
    return [];
}

function isScenarioActive(name) {
    if (activeStrategyScenario === name) return true;
    if (!currentStints || currentStints.length === 0) return false;
    if (name === 'ham_no_stop') {
        return currentStints.length === 2 &&
               currentStints[0].compound === 'MEDIUM' &&
               currentStints[0].lap_end === 20 &&
               currentStints[1].compound === 'HARD';
    } else if (name === 'norris_2stop') {
        return currentStints.length === 3 &&
               currentStints[0].compound === 'MEDIUM' &&
               currentStints[0].lap_end === 20 &&
               currentStints[1].compound === 'HARD' &&
               currentStints[1].lap_end === 44 &&
               currentStints[2].compound === 'SOFT';
    } else if (name === 'sainz_early') {
        return currentStints.length === 2 &&
               currentStints[0].compound === 'MEDIUM' &&
               currentStints[0].lap_end === 15 &&
               currentStints[1].compound === 'HARD';
    }
    return false;
}

// Initialize application on page load with multi-page awareness
document.addEventListener('DOMContentLoaded', async () => {
    // Only run race studio logic if on the Race Studio page
    if (document.getElementById('driver-select')) {
        await loadInitialData();
    }
    // Only run validation logic if on the Validation page
    if (document.getElementById('replay-table-body')) {
        await loadValidationData();
    }
});

async function loadInitialData() {
    try {
        const res = await fetch('/api/race-info');
        const data = await res.json();
        raceInfo = data.race_info;
        baselineStints = data.baseline_stints;
        officialClassification = data.official_classification;

        // Parse URL query parameters for cross-page deep linking
        const params = new URLSearchParams(window.location.search);

        if (params.get('reset') === '1') {
            resetToBaseline(true);
            return;
        }

        if (params.has('driver')) {
            const dNum = parseInt(params.get('driver'));
            if (baselineStints[dNum]) {
                selectedDriver = dNum;
            }
        }

        populateDriverDropdown();
        currentStints = JSON.parse(JSON.stringify(baselineStints[selectedDriver] || []));

        // Options toggles from URL params
        if (params.get('traffic') === '1') {
            const el = document.getElementById('toggle-traffic-model');
            if (el) el.checked = true;
        }
        if (params.get('tyre') === '1') {
            const el = document.getElementById('toggle-nonlinear-tyre');
            if (el) el.checked = true;
        }
        if (params.get('remove_sc') === '1') {
            const el = document.getElementById('toggle-remove-sc');
            if (el) el.checked = true;
        }
        if (params.get('remove_vsc') === '1') {
            const el = document.getElementById('toggle-remove-vsc');
            if (el) el.checked = true;
        }

        if (params.has('preset')) {
            loadPreset(params.get('preset'));
            return;
        }

        renderStintRows();
        updatePresetButtonStyles();
        await runSimulation();

        // If a natural language prompt was passed from ML Lab or elsewhere:
        if (params.has('prompt')) {
            const promptVal = params.get('prompt');
            const chatInput = document.getElementById('ai-chat-input');
            if (chatInput && promptVal) {
                chatInput.value = promptVal;
                setTimeout(() => {
                    handleChatSubmit(new Event('submit'));
                }, 350);
            }
        }
    } catch (err) {
        console.error('Failed to load initial race data:', err);
    }
}

// Natural Language AI Strategy Assistant Handler
function fillChatInput(text) {
    const input = document.getElementById('ai-chat-input');
    input.value = text;
    handleChatSubmit(new Event('submit'));
}

async function handleChatSubmit(event) {
    if (event) event.preventDefault();
    const input = document.getElementById('ai-chat-input');
    const prompt = input.value.trim();
    if (!prompt) return;

    const responseBox = document.getElementById('ai-chat-response-box');
    const responseText = document.getElementById('ai-chat-response-text');
    const badge = document.getElementById('ai-action-badge');
    const submitBtn = document.getElementById('chat-submit-btn');

    // UI Loading State
    responseBox.classList.remove('hidden');
    responseText.innerHTML = '<span class="text-gray-400 italic flex items-center gap-2"><svg class="animate-spin h-3.5 w-3.5 text-f1red" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg> Parsing prompt and re-calculating race strategy...</span>';
    badge.textContent = 'Processing';
    badge.className = 'text-[10px] bg-gray-800 text-gray-300 border border-gray-700 px-2 py-0.5 rounded font-mono font-semibold';

    const removeSC = document.getElementById('toggle-remove-sc')?.checked || false;
    const removeVSC = document.getElementById('toggle-remove-vsc')?.checked || false;
    const insertSC = document.getElementById('toggle-insert-sc')?.checked || false;
    const insertSCLap = insertSC ? parseInt(document.getElementById('insert-sc-lap').value) : null;
    const insertVSC = document.getElementById('toggle-insert-vsc')?.checked || false;
    const insertVSCLap = insertVSC ? parseInt(document.getElementById('insert-vsc-lap').value) : null;
    const replaceSlowStop = document.getElementById('toggle-replace-slow-stop')?.checked || false;
    const enableTraffic = document.getElementById('toggle-traffic-model')?.checked || false;
    const useNonlinearTyre = document.getElementById('toggle-nonlinear-tyre')?.checked || false;

    const currentState = {
        modified_driver: selectedDriver,
        custom_stints: currentStints,
        remove_safety_car: removeSC,
        remove_vsc: removeVSC,
        insert_safety_car_lap: insertSCLap,
        insert_vsc_lap: insertVSCLap,
        replace_slow_stop: replaceSlowStop,
        enable_traffic_model: enableTraffic,
        use_nonlinear_tyre: useNonlinearTyre
    };

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt, current_state: currentState })
        });
        const data = await res.json();

        // Convert simple markdown in response to HTML with clickable links
        let formatted = (data.response || '')
            .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>')
            .replace(/\*(.*?)\*/g, '<em class="text-gray-300">$1</em>')
            .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" class="text-cyan-400 hover:text-cyan-300 underline font-semibold transition">$1 ↗</a>')
            .replace(/\n/g, '<br>');

        responseText.innerHTML = formatted;

        const confBadge = document.getElementById('ai-confidence-badge');
        if (confBadge && data.confidence !== undefined) {
            confBadge.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
            confBadge.classList.remove('hidden');
        }

        if (data.type === 'COMMAND_EXECUTED') {
            badge.textContent = `Action: ${data.action.replaceAll('_', ' ')}`;
            badge.className = 'text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-800 px-2 py-0.5 rounded font-mono font-semibold';

            // Sync UI state if simulation was modified
            const directives = data.ui_directives || data.state;
            if (directives) {
                if (directives.modified_driver) {
                    selectedDriver = directives.modified_driver;
                    const dSelect = document.getElementById('driver-select');
                    if (dSelect) dSelect.value = selectedDriver;
                    updateDriverTeamLabel();
                }
                if (directives.custom_stints) {
                    currentStints = directives.custom_stints;
                    activeStrategyScenario = null;
                    renderStintRows();
                }
                if (directives.enable_traffic_model !== undefined && document.getElementById('toggle-traffic-model')) {
                    document.getElementById('toggle-traffic-model').checked = directives.enable_traffic_model;
                }
                if (directives.use_nonlinear_tyre !== undefined && document.getElementById('toggle-nonlinear-tyre')) {
                    document.getElementById('toggle-nonlinear-tyre').checked = directives.use_nonlinear_tyre;
                }
                if (directives.remove_safety_car !== undefined && document.getElementById('toggle-remove-sc')) {
                    document.getElementById('toggle-remove-sc').checked = directives.remove_safety_car;
                }
                if (directives.remove_vsc !== undefined && document.getElementById('toggle-remove-vsc')) {
                    document.getElementById('toggle-remove-vsc').checked = directives.remove_vsc;
                }
                if (directives.insert_safety_car_lap && document.getElementById('toggle-insert-sc')) {
                    document.getElementById('toggle-insert-sc').checked = true;
                    document.getElementById('insert-sc-controls').classList.remove('hidden');
                    document.getElementById('insert-sc-lap').value = directives.insert_safety_car_lap;
                    document.getElementById('insert-sc-lap-val').innerText = directives.insert_safety_car_lap;
                }
                if (directives.insert_vsc_lap && document.getElementById('toggle-insert-vsc')) {
                    document.getElementById('toggle-insert-vsc').checked = true;
                    document.getElementById('insert-vsc-controls').classList.remove('hidden');
                    document.getElementById('insert-vsc-lap').value = directives.insert_vsc_lap;
                    document.getElementById('insert-vsc-lap-val').innerText = directives.insert_vsc_lap;
                }
                if (directives.replace_slow_stop !== undefined && document.getElementById('toggle-replace-slow-stop')) {
                    document.getElementById('toggle-replace-slow-stop').checked = directives.replace_slow_stop;
                }
                updatePresetButtonStyles();
            }

            if (data.simulation_result) {
                renderResults(data.simulation_result);
                setTimeout(() => {
                    const resultsSection = document.getElementById('finishing-order-section');
                    if (resultsSection) resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 200);
            }
        } else {
            badge.textContent = 'Q&A Answer';
            badge.className = 'text-[10px] bg-blue-950 text-blue-400 border border-blue-800 px-2 py-0.5 rounded font-mono font-semibold';
        }
    } catch (err) {
        console.error('Chat error:', err);
        responseText.innerHTML = '<span class="text-rose-400">Failed to communicate with strategy assistant.</span>';
        badge.textContent = 'Error';
        badge.className = 'text-[10px] bg-rose-950 text-rose-400 border border-rose-800 px-2 py-0.5 rounded font-mono font-semibold';
    }
}

function populateDriverDropdown() {
    const sel = document.getElementById('driver-select');
    sel.innerHTML = '';
    officialClassification.forEach(d => {
        const opt = document.createElement('option');
        opt.value = d.driver_number;
        opt.textContent = `P${d.pos} • ${d.name} (${d.code}) — ${d.team}`;
        if (d.driver_number === selectedDriver) opt.selected = true;
        sel.appendChild(opt);
    });
    updateDriverTeamLabel();
}

function updateDriverTeamLabel() {
    const d = officialClassification.find(x => x.driver_number === selectedDriver);
    if (d) {
        const teamEl = document.getElementById('selected-driver-team');
        if (teamEl) {
            teamEl.textContent = `${d.team} • #${d.driver_number}`;
        }
        const mlLink = document.getElementById('link-inspect-driver-ml');
        if (mlLink) {
            mlLink.href = `/ml-lab?driver=${selectedDriver}`;
            mlLink.setAttribute('title', `Inspect ${d.name}'s telemetry in ML Lab`);
        }
    }
}

function onDriverChange() {
    selectedDriver = parseInt(document.getElementById('driver-select').value);
    localStorage.setItem('f1_selected_driver', selectedDriver);
    updateDriverTeamLabel();

    // Check if any scenario preset is active
    const activeScenario = activeStrategyScenario ||
        (isScenarioActive('ham_no_stop') ? 'ham_no_stop' :
         isScenarioActive('norris_2stop') ? 'norris_2stop' :
         isScenarioActive('sainz_early') ? 'sainz_early' : null);

    // If a strategy scenario preset is active, keep it active and apply that scenario to the newly selected driver!
    if (activeScenario) {
        activeStrategyScenario = activeScenario;
        currentStints = getStrategyStintsForScenario(activeScenario);
        renderStintRows();
        runSimulation();
        updatePresetButtonStyles();
        return;
    }

    // If user has set a multi-stop strategy (e.g. 2-stop or 3-stop), preserve that stop count for the new driver
    const prevStintCount = currentStints ? currentStints.length : 2;
    if (prevStintCount > 2) {
        setStopCount(prevStintCount - 1);
    } else {
        currentStints = JSON.parse(JSON.stringify(baselineStints[selectedDriver] || []));
        renderStintRows();
        runSimulation();
    }
    updatePresetButtonStyles();
}

function setStopCount(count) {
    activeStrategyScenario = null;
    activePreset = 'custom';
    let pitLaps = [];
    let compounds = [];
    if (count === 1) {
        pitLaps = [24];
        compounds = ['MEDIUM', 'HARD'];
    } else if (count === 2) {
        pitLaps = [20, 42];
        compounds = ['MEDIUM', 'HARD', 'MEDIUM'];
    } else if (count === 3) {
        pitLaps = [16, 32, 48];
        compounds = ['SOFT', 'MEDIUM', 'HARD', 'SOFT'];
    }

    const numStints = count + 1;
    currentStints = [];
    let startLap = 1;
    for (let i = 0; i < count; i++) {
        currentStints.push({
            stint_number: i + 1,
            lap_start: startLap,
            lap_end: pitLaps[i],
            compound: compounds[i],
            start_tyre_age: 0
        });
        startLap = pitLaps[i] + 1;
    }
    currentStints.push({
        stint_number: numStints,
        lap_start: startLap,
        lap_end: raceInfo.total_laps || 62,
        compound: compounds[numStints - 1],
        start_tyre_age: 0
    });

    renderStintRows();
    updatePresetButtonStyles();
    runSimulation();
}

let latestCliffAnalysis = null;

function renderStintRows(cliffAnalysis = null) {
    if (cliffAnalysis) {
        latestCliffAnalysis = cliffAnalysis;
    }
    const container = document.getElementById('stint-rows-container');
    container.innerHTML = '';
    const numStints = currentStints.length;
    document.getElementById('stint-count-label').textContent = `${numStints} Stint${numStints > 1 ? 's' : ''} (${numStints - 1} Pit Stop${numStints - 1 > 1 ? 's' : ''})`;

    // Update stop buttons active state
    [1, 2, 3].forEach(c => {
        const btn = document.getElementById(`stop-btn-${c}`);
        if (btn) {
            if (numStints === c + 1) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        }
    });

    const cliffThresholds = { 'SOFT': 18, 'MEDIUM': 28, 'HARD': 42 };

    currentStints.forEach((s, idx) => {
        const isLast = idx === numStints - 1;
        const stintLaps = s.lap_end - s.lap_start + 1;
        const row = document.createElement('div');
        row.className = 'bg-gray-800/70 p-2.5 rounded border border-gray-700/80 space-y-2';

        const header = document.createElement('div');
        header.className = 'flex items-center justify-between text-xs font-semibold';
        header.innerHTML = `
            <span class="flex items-center gap-1.5">
                <span class="w-1.5 h-1.5 rounded-full bg-gray-400"></span>
                Stint ${s.stint_number}: Laps ${s.lap_start}–${s.lap_end}
                <span class="text-[10px] text-gray-400 font-normal">(${stintLaps} laps)</span>
            </span>
            <span class="tyre-badge tyre-${s.compound.toLowerCase()}">${s.compound}</span>
        `;
        row.appendChild(header);

        // Control Row: Compound Picker & Pit Lap Slider
        const controls = document.createElement('div');
        controls.className = 'grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs pt-1';

        // Compound select
        const compDiv = document.createElement('div');
        compDiv.className = 'flex items-center gap-1.5';
        compDiv.innerHTML = `
            <span class="text-gray-400 text-[11px]">Tyre:</span>
            <select onchange="onStintCompoundChange(${idx}, this.value)" class="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs text-white font-semibold">
                <option value="SOFT" ${s.compound === 'SOFT' ? 'selected' : ''}>🔴 SOFT</option>
                <option value="MEDIUM" ${s.compound === 'MEDIUM' ? 'selected' : ''}>🟡 MEDIUM</option>
                <option value="HARD" ${s.compound === 'HARD' ? 'selected' : ''}>⚪ HARD</option>
            </select>
        `;
        controls.appendChild(compDiv);

        // Pit lap slider (if not final stint)
        if (!isLast) {
            const pitDiv = document.createElement('div');
            pitDiv.className = 'flex items-center gap-2';
            const minPit = s.lap_start + 1;
            const maxPit = currentStints[idx + 1].lap_end - 1;
            pitDiv.innerHTML = `
                <span class="text-gray-400 text-[11px] whitespace-nowrap">Pit Lap:</span>
                <input type="range" min="${minPit}" max="${maxPit}" value="${s.lap_end}"
                    oninput="onStintPitLapInput(${idx}, this.value)"
                    onchange="runSimulation()"
                    class="w-full accent-f1red">
                <span class="font-mono font-bold text-white text-[11px] w-6 text-right" id="pit-label-${idx}">${s.lap_end}</span>
            `;
            controls.appendChild(pitDiv);
        }

        row.appendChild(controls);

        // Model 2: Non-Linear Tyre Cliff Warning & Status Badge
        const cliffLimit = cliffThresholds[s.compound] || 30;
        let cliffAnalysisItem = (latestCliffAnalysis && latestCliffAnalysis[idx]) ? latestCliffAnalysis[idx] : null;

        let statusLabel = `🟢 Optimal Wear (${stintLaps}L / Max ${cliffLimit}L)`;
        let badgeClass = 'bg-emerald-950 text-emerald-400 border-emerald-800';

        if (cliffAnalysisItem) {
            statusLabel = cliffAnalysisItem.status_label;
            if (cliffAnalysisItem.health_status === 'NEAR_CLIFF') {
                badgeClass = 'bg-amber-950 text-amber-400 border-amber-800';
            } else if (cliffAnalysisItem.health_status === 'EXCEEDED_CLIFF') {
                badgeClass = 'bg-rose-950 text-rose-400 border-rose-800';
            }
        } else {
            if (stintLaps > cliffLimit) {
                statusLabel = `🔴 Exceeded Thermal Cliff (+${((stintLaps - cliffLimit) * 0.08).toFixed(2)}s/lap)`;
                badgeClass = 'bg-rose-950 text-rose-400 border-rose-800';
            } else if (stintLaps >= cliffLimit - 3) {
                statusLabel = `🟡 Near Cliff (${cliffLimit - stintLaps} laps left)`;
                badgeClass = 'bg-amber-950 text-amber-400 border-amber-800';
            }
        }

        const cliffDiv = document.createElement('div');
        cliffDiv.className = 'flex items-center justify-between text-[11px] font-mono pt-1.5 border-t border-gray-700/50';
        cliffDiv.innerHTML = `
            <span class="px-2 py-0.5 rounded border ${badgeClass} text-[10px] font-semibold flex items-center gap-1">
                <span>${statusLabel}</span>
            </span>
            <span class="text-gray-400 text-[10px]" title="Model 2: HistGradientBoosting Non-Linear Degradation">M2 Cliff: L${cliffLimit}</span>
        `;
        row.appendChild(cliffDiv);

        container.appendChild(row);
    });
}

function onStintCompoundChange(idx, compound) {
    currentStints[idx].compound = compound;
    activePreset = 'custom';
    activeStrategyScenario = null;
    updatePresetButtonStyles();
    renderStintRows();
    runSimulation();
}

function onStintPitLapInput(idx, newLap) {
    const val = parseInt(newLap);
    document.getElementById(`pit-label-${idx}`).textContent = val;
    currentStints[idx].lap_end = val;
    if (idx < currentStints.length - 1) {
        currentStints[idx + 1].lap_start = val + 1;
    }
    activeStrategyScenario = null;
    activePreset = 'custom';
    updatePresetButtonStyles();
}

function onOptionToggle() {
    updatePresetButtonStyles();
    runSimulation();
}

function onInsertSCToggle() {
    const isChecked = document.getElementById('toggle-insert-sc').checked;
    const controls = document.getElementById('insert-sc-controls');
    if (isChecked) {
        controls.classList.remove('hidden');
    } else {
        controls.classList.add('hidden');
    }
    updatePresetButtonStyles();
    runSimulation();
}

function onInsertVSCToggle() {
    const isChecked = document.getElementById('toggle-insert-vsc').checked;
    const controls = document.getElementById('insert-vsc-controls');
    if (isChecked) {
        controls.classList.remove('hidden');
    } else {
        controls.classList.add('hidden');
    }
    updatePresetButtonStyles();
    runSimulation();
}

function toggleAIInfoModal() {
    const drawer = document.getElementById('ai-info-drawer');
    if (drawer) {
        drawer.classList.toggle('hidden');
    }
}

function resetDriverStrategy() {
    activeStrategyScenario = null;
    currentStints = JSON.parse(JSON.stringify(baselineStints[selectedDriver] || []));
    renderStintRows();
    updatePresetButtonStyles();
    runSimulation();
}

function resetToBaseline(keepML = true) {
    activeStrategyScenario = null;
    // Preserve active ML Model settings if keepML is true
    const enableTraffic = keepML ? (document.getElementById('toggle-traffic-model')?.checked || false) : false;
    const useNonlinear = keepML ? (document.getElementById('toggle-nonlinear-tyre')?.checked || false) : false;

    // Reset All Intervention Toggles
    if (document.getElementById('toggle-remove-sc')) document.getElementById('toggle-remove-sc').checked = false;
    if (document.getElementById('toggle-remove-vsc')) document.getElementById('toggle-remove-vsc').checked = false;
    if (document.getElementById('toggle-insert-sc')) document.getElementById('toggle-insert-sc').checked = false;
    if (document.getElementById('toggle-insert-vsc')) document.getElementById('toggle-insert-vsc').checked = false;
    if (document.getElementById('toggle-replace-slow-stop')) document.getElementById('toggle-replace-slow-stop').checked = false;

    // Preserve ML Model selections
    if (document.getElementById('toggle-traffic-model')) document.getElementById('toggle-traffic-model').checked = enableTraffic;
    if (document.getElementById('toggle-nonlinear-tyre')) document.getElementById('toggle-nonlinear-tyre').checked = useNonlinear;

    // Hide insertion sliders
    if (document.getElementById('insert-sc-controls')) document.getElementById('insert-sc-controls').classList.add('hidden');
    if (document.getElementById('insert-vsc-controls')) document.getElementById('insert-vsc-controls').classList.add('hidden');

    // Reset AI Assistant Chat output
    const chatBox = document.getElementById('ai-chat-response-box');
    if (chatBox) chatBox.classList.add('hidden');
    const chatInput = document.getElementById('ai-chat-input');
    if (chatInput) chatInput.value = '';

    // Reset Driver & Stints
    selectedDriver = 55; // Sainz
    const dSelect = document.getElementById('driver-select');
    if (dSelect) dSelect.value = 55;
    updateDriverTeamLabel();

    currentStints = JSON.parse(JSON.stringify(baselineStints[selectedDriver] || []));
    renderStintRows();
    updatePresetButtonStyles();
    runSimulation();
}

function loadPreset(name) {
    if (name === 'baseline') {
        activeStrategyScenario = null;
        resetToBaseline(true);
        return;
    }

    if (name === 'no_sc') {
        const scToggle = document.getElementById('toggle-remove-sc');
        if (scToggle) {
            scToggle.checked = !scToggle.checked;
        }
        updatePresetButtonStyles();
        runSimulation();
        return;
    }

    if (name === 'no_vsc') {
        const vscToggle = document.getElementById('toggle-remove-vsc');
        if (vscToggle) {
            vscToggle.checked = !vscToggle.checked;
        }
        updatePresetButtonStyles();
        runSimulation();
        return;
    }

    // Toggle off if clicking an already active scenario preset
    if (activeStrategyScenario === name || isScenarioActive(name)) {
        activeStrategyScenario = null;
        currentStints = JSON.parse(JSON.stringify(baselineStints[selectedDriver] || []));
        renderStintRows();
        updatePresetButtonStyles();
        runSimulation();
        return;
    }

    // Activating strategy scenario preset
    activeStrategyScenario = name;
    if (name === 'ham_no_stop') {
        selectedDriver = 44;
    } else if (name === 'norris_2stop') {
        selectedDriver = 4;
    } else if (name === 'sainz_early') {
        selectedDriver = 55;
    }

    localStorage.setItem('f1_selected_driver', selectedDriver);
    const dSelect = document.getElementById('driver-select');
    if (dSelect) dSelect.value = selectedDriver;

    currentStints = getStrategyStintsForScenario(name);
    updateDriverTeamLabel();
    renderStintRows();
    updatePresetButtonStyles();
    runSimulation();
}

function updatePresetButtonStyles() {
    // 1. SC and VSC preset pills reflect the actual checkbox state
    const isScRemoved = document.getElementById('toggle-remove-sc')?.checked || false;
    const isVscRemoved = document.getElementById('toggle-remove-vsc')?.checked || false;

    const scBtn = document.getElementById('btn-preset-no_sc');
    if (scBtn) {
        if (isScRemoved) scBtn.classList.add('active');
        else scBtn.classList.remove('active');
    }

    const vscBtn = document.getElementById('btn-preset-no_vsc');
    if (vscBtn) {
        if (isVscRemoved) vscBtn.classList.add('active');
        else vscBtn.classList.remove('active');
    }

    // Driver information for dynamic label updates
    const dObj = officialClassification ? officialClassification.find(x => x.driver_number === selectedDriver) : null;
    const dCode = dObj ? dObj.code : '';

    // 2. Driver Scenario Presets reflect active scenario (supports any driver)
    const hamBtn = document.getElementById('btn-preset-ham_no_stop');
    const isHamActive = isScenarioActive('ham_no_stop');
    if (hamBtn) {
        if (isHamActive) hamBtn.classList.add('active');
        else hamBtn.classList.remove('active');
    }
    const labelHam = document.getElementById('label-preset-ham');
    if (labelHam) {
        if (isHamActive && selectedDriver !== 44 && selectedDriver !== 63) {
            labelHam.innerHTML = `🛡️ ${dCode} 1-Stop`;
        } else {
            labelHam.innerHTML = `🛡️ Mercedes 1-Stop`;
        }
    }

    const norrisBtn = document.getElementById('btn-preset-norris_2stop');
    const isNorrisActive = isScenarioActive('norris_2stop');
    if (norrisBtn) {
        if (isNorrisActive) norrisBtn.classList.add('active');
        else norrisBtn.classList.remove('active');
    }
    const labelNorris = document.getElementById('label-preset-norris');
    if (labelNorris) {
        if (isNorrisActive && selectedDriver !== 4) {
            labelNorris.innerHTML = `⚡ ${dCode} 2-Stop Soft`;
        } else {
            labelNorris.innerHTML = `⚡ Norris 2-Stop Soft`;
        }
    }

    const sainzBtn = document.getElementById('btn-preset-sainz_early');
    const isSainzActive = isScenarioActive('sainz_early');
    if (sainzBtn) {
        if (isSainzActive) sainzBtn.classList.add('active');
        else sainzBtn.classList.remove('active');
    }
    const labelSainz = document.getElementById('label-preset-sainz');
    if (labelSainz) {
        if (isSainzActive && selectedDriver !== 55) {
            labelSainz.innerHTML = `⏱️ ${dCode} Pit Lap 15`;
        } else {
            labelSainz.innerHTML = `⏱️ Sainz Pit Lap 15`;
        }
    }

    // 3. Baseline Button is active only when in baseline condition
    const baseBtn = document.getElementById('btn-preset-baseline');
    if (baseBtn) {
        const isBaseDriver = selectedDriver === 55;
        const isBaseStints = JSON.stringify(currentStints) === JSON.stringify(baselineStints[55] || []);
        const noInterventions = !isScRemoved && !isVscRemoved &&
            !(document.getElementById('toggle-insert-sc')?.checked) &&
            !(document.getElementById('toggle-insert-vsc')?.checked) &&
            !(document.getElementById('toggle-replace-slow-stop')?.checked);
        if (isBaseDriver && isBaseStints && noInterventions && !activeStrategyScenario && !isHamActive && !isNorrisActive && !isSainzActive) {
            baseBtn.classList.add('active');
        } else {
            baseBtn.classList.remove('active');
        }
    }
}

window.handleGlobalReset = function() {
    resetToBaseline(true);
};

async function runSimulation() {
    const removeSC = document.getElementById('toggle-remove-sc')?.checked || false;
    const removeVSC = document.getElementById('toggle-remove-vsc')?.checked || false;
    const insertSC = document.getElementById('toggle-insert-sc')?.checked || false;
    const insertSCLap = insertSC ? parseInt(document.getElementById('insert-sc-lap').value) : null;
    const insertVSC = document.getElementById('toggle-insert-vsc')?.checked || false;
    const insertVSCLap = insertVSC ? parseInt(document.getElementById('insert-vsc-lap').value) : null;
    const replaceSlowStop = document.getElementById('toggle-replace-slow-stop')?.checked || false;
    const enableTraffic = document.getElementById('toggle-traffic-model')?.checked || false;
    const useNonlinearTyre = document.getElementById('toggle-nonlinear-tyre')?.checked || false;

    const payload = {
        modified_driver: selectedDriver,
        custom_stints: currentStints,
        remove_safety_car: removeSC,
        remove_vsc: removeVSC,
        insert_safety_car_lap: insertSCLap,
        insert_safety_car_duration: 3,
        insert_vsc_lap: insertVSCLap,
        insert_vsc_duration: 2,
        replace_slow_stop: replaceSlowStop,
        enable_traffic_model: enableTraffic,
        use_nonlinear_tyre: useNonlinearTyre
    };

    try {
        const res = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        renderResults(data);
    } catch (err) {
        console.error('Simulation execution failed:', err);
    }
}

function renderResults(data) {
    // 0. Update Cliff Analysis & Stint Rows if present
    if (data.stint_cliff_analysis) {
        latestCliffAnalysis = data.stint_cliff_analysis;
        renderStintRows(data.stint_cliff_analysis);
    }

    // 1. Render Table
    const tbody = document.getElementById('results-table-body');
    tbody.innerHTML = '';
    const enableTraffic = document.getElementById('toggle-traffic-model')?.checked || false;

    data.comparison.forEach(r => {
        const tr = document.createElement('tr');
        if (r.is_modified) {
            tr.className = 'row-modified';
        } else if (r.sim_position === 1) {
            tr.className = 'row-winner';
        }

        let deltaHtml = '<span class="delta-same">-</span>';
        if (r.position_delta > 0) {
            deltaHtml = `<span class="delta-gain">▲ +${r.position_delta}</span>`;
        } else if (r.position_delta < 0) {
            deltaHtml = `<span class="delta-loss">▼ ${r.position_delta}</span>`;
        }

        const gapDisplay = r.sim_position === 1 ? '<span class="text-emerald-400 font-bold">Leader</span>' : `+${r.sim_gap_to_leader.toFixed(3)}s`;

        // Model 1: Driver Traffic & Dirty Air Display
        let trafHtml = '<span class="text-gray-500 font-mono text-[10px]">Clean air</span>';
        if (!enableTraffic) {
            trafHtml = `<a href="/ml-lab?model=traffic" class="text-gray-500 hover:text-cyan-400 font-mono text-[10px] transition" title="Click to test Overtake Classifier in ML Lab">Clean air (Off) ↗</a>`;
        } else if (r.traffic_summary && r.traffic_summary.laps_in_dirty_air > 0) {
            const rival = r.traffic_summary.trapped_behind ? `behind ${r.traffic_summary.trapped_behind.code}` : 'dirty air';
            trafHtml = `<a href="/ml-lab?driver=${r.driver_number}&model=traffic" class="text-amber-400 font-mono text-[10px] font-semibold bg-amber-950/70 hover:bg-amber-900/80 px-1.5 py-0.5 rounded border border-amber-900/60 inline-flex items-center gap-1 transition" title="${r.traffic_summary.laps_in_dirty_air} laps in dirty air, lost ${r.traffic_summary.time_lost_dirty_air}s. Click to inspect in ML Lab.">
                <span>⚠ ${r.traffic_summary.laps_in_dirty_air}L ${rival} (-${r.traffic_summary.time_lost_dirty_air}s)</span>
                <span class="text-[9px] opacity-75">↗</span>
            </a>`;
        } else if (r.traffic_summary && r.traffic_summary.overtakes_made > 0) {
            trafHtml = `<a href="/ml-lab?driver=${r.driver_number}&model=traffic" class="text-emerald-400 font-mono text-[10px] font-semibold bg-emerald-950/70 hover:bg-emerald-900/80 px-1.5 py-0.5 rounded border border-emerald-900/60 inline-flex items-center gap-1 transition" title="${r.traffic_summary.overtakes_made} overtake(s) completed. Click to inspect in ML Lab.">
                <span>✓ ${r.traffic_summary.overtakes_made} Overtake${r.traffic_summary.overtakes_made > 1 ? 's' : ''}</span>
                <span class="text-[9px] opacity-75">↗</span>
            </a>`;
        }

        tr.innerHTML = `
            <td class="py-2.5 px-2 font-bold ${r.sim_position === 1 ? 'text-emerald-400' : 'text-white'}">P${r.sim_position}</td>
            <td class="py-2.5 px-2 text-gray-400">P${r.actual_position}</td>
            <td class="py-2.5 px-2">${deltaHtml}</td>
            <td class="py-2.5 px-2 font-bold text-white flex items-center gap-1.5">
                <span>${r.code}</span>
                ${r.is_modified ? '<span class="bg-amber-400 text-black text-[9px] font-black px-1 rounded uppercase">MOD</span>' : ''}
            </td>
            <td class="py-2.5 px-2 text-gray-300 font-sans text-[11px]">${r.team}</td>
            <td class="py-2.5 px-2 text-gray-400 text-[11px]">${r.strategy}</td>
            <td class="py-2.5 px-2 font-mono">${trafHtml}</td>
            <td class="py-2.5 px-2 text-right font-mono">${gapDisplay}</td>
            <td class="py-2.5 px-2 text-right text-gray-400 font-mono text-[11px]">${r.actual_official_time}</td>
        `;
        tbody.appendChild(tr);
    });

    // 1.5 Update Battles & Dirty Air Telemetry Card (Model 1)
    const trafStatus = document.getElementById('traffic-status-indicator');
    if (trafStatus) {
        if (enableTraffic) {
            trafStatus.textContent = 'ML Traffic Active (Logistic Reg)';
            trafStatus.className = 'text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800';
        } else {
            trafStatus.textContent = 'Traffic Off (Clean Air)';
            trafStatus.className = 'text-[10px] font-mono px-2 py-0.5 rounded bg-gray-900 text-gray-500 border border-gray-800';
        }
    }

    const banner = document.getElementById('driver-traffic-banner');
    if (banner && data.driver_traffic_summary) {
        const selTraf = data.driver_traffic_summary[selectedDriver] || { laps_in_dirty_air: 0, time_lost_dirty_air: 0, overtakes_made: 0, overtakes_blocked: 0 };
        const rivalText = selTraf.trapped_behind ? `Stuck behind ${selTraf.trapped_behind.code}` : (selTraf.laps_in_dirty_air > 0 ? 'Dirty Air Turbulence' : 'Clean Track Air');

        banner.innerHTML = `
            <div class="bg-gray-950/70 p-3 rounded-xl border border-gray-800 space-y-1">
                <span class="text-[10px] text-gray-400 uppercase tracking-wider block font-sans">Active Driver Status</span>
                <div class="text-sm font-bold text-white flex items-center gap-1.5">
                    <span class="${selTraf.laps_in_dirty_air > 0 ? 'text-amber-400' : 'text-emerald-400'}">${rivalText}</span>
                </div>
            </div>
            <div class="bg-gray-950/70 p-3 rounded-xl border border-gray-800 space-y-1">
                <span class="text-[10px] text-gray-400 uppercase tracking-wider block font-sans">Time Lost to Dirty Air</span>
                <div class="text-sm font-bold ${selTraf.time_lost_dirty_air > 0 ? 'text-rose-400' : 'text-gray-300'}">
                    +${selTraf.time_lost_dirty_air.toFixed(2)}s <span class="text-[11px] text-gray-500 font-normal">(${selTraf.laps_in_dirty_air} laps)</span>
                </div>
            </div>
            <div class="bg-gray-950/70 p-3 rounded-xl border border-gray-800 space-y-1">
                <span class="text-[10px] text-gray-400 uppercase tracking-wider block font-sans">Overtakes & Blocks</span>
                <div class="text-sm font-bold text-cyan-400">
                    ${selTraf.overtakes_made} Made • <span class="text-amber-400">${selTraf.overtakes_blocked} Blocked</span>
                </div>
            </div>
        `;
    }

    const feed = document.getElementById('traffic-events-feed');
    if (feed) {
        if (!enableTraffic) {
            feed.innerHTML = '<div class="text-gray-500 italic py-1">Traffic model is disabled. All drivers run at unobstructed flying pace in clean air.</div>';
        } else if (!data.traffic_events || data.traffic_events.length === 0) {
            feed.innerHTML = '<div class="text-gray-400 italic py-1">No proximity battles or dirty air bottlenecks occurred under this scenario.</div>';
        } else {
            feed.innerHTML = '';
            data.traffic_events.slice(-10).reverse().forEach(ev => {
                const item = document.createElement('div');
                item.className = 'flex items-center justify-between py-1 border-b border-gray-900/60 last:border-0';
                if (ev.type === 'DIRTY_AIR_BLOCK') {
                    item.innerHTML = `
                        <div class="flex items-center gap-2">
                            <span class="text-amber-400 font-bold">L${ev.lap}</span>
                            <span class="text-gray-300"><strong>${ev.chaser}</strong> trapped behind <strong>${ev.leader}</strong></span>
                            <span class="text-[10px] text-gray-500 font-mono">(Gap: ${ev.gap}s)</span>
                        </div>
                        <span class="text-[10px] bg-rose-950/80 text-rose-300 border border-rose-800 px-1.5 py-0.5 rounded font-mono">
                            Blocked in Dirty Air (P=${(ev.prob * 100).toFixed(0)}%)
                        </span>
                    `;
                } else {
                    item.innerHTML = `
                        <div class="flex items-center gap-2">
                            <span class="text-emerald-400 font-bold">L${ev.lap}</span>
                            <span class="text-gray-300"><strong>${ev.chaser}</strong> passed <strong>${ev.leader}</strong></span>
                            <span class="text-[10px] text-gray-500 font-mono">(Gap: ${ev.gap}s)</span>
                        </div>
                        <span class="text-[10px] bg-emerald-950/80 text-emerald-300 border border-emerald-800 px-1.5 py-0.5 rounded font-mono">
                            Overtake Successful (P=${(ev.prob * 100).toFixed(0)}%)
                        </span>
                    `;
                }
                feed.appendChild(item);
            });
        }
    }

    // 2. Render Plotly Chart
    renderGapChart(data.chart_data);

    // 3. Render Explanation
    const explContainer = document.getElementById('explanation-text');
    explContainer.innerHTML = '';

    const summaryP = document.createElement('p');
    summaryP.className = 'font-medium text-white leading-relaxed text-xs';
    const formattedSummary = (data.explanation.summary || '')
        .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>')
        .replace(/\*(.*?)\*/g, '<em class="text-gray-300">$1</em>')
        .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" class="text-cyan-400 hover:text-cyan-300 underline font-semibold transition">$1 ↗</a>');
    summaryP.innerHTML = formattedSummary;
    explContainer.appendChild(summaryP);

    if (data.explanation.assumptions_summary && data.explanation.assumptions_summary.length > 0) {
        const notesUl = document.createElement('ul');
        notesUl.className = 'list-disc pl-4 space-y-1 text-gray-400 text-[11px] mt-2';
        data.explanation.assumptions_summary.forEach(note => {
            const li = document.createElement('li');
            li.textContent = note;
            notesUl.appendChild(li);
        });
        explContainer.appendChild(notesUl);
    }

    // 4. Update Active Assumptions Cards
    const models = data.active_assumptions.models;
    if (models) {
        const pace = models.pace_and_degradation;
        const pit = models.pit_loss;
        const track = models.track_conditions;

        if (pace && pace.compound_degradation_sec_per_lap) {
            document.getElementById('param-deg-soft').textContent = `+${pace.compound_degradation_sec_per_lap.SOFT.toFixed(3)} s/lap`;
            document.getElementById('param-deg-med').textContent = `+${pace.compound_degradation_sec_per_lap.MEDIUM.toFixed(3)} s/lap`;
            document.getElementById('param-deg-hard').textContent = `+${pace.compound_degradation_sec_per_lap.HARD.toFixed(3)} s/lap`;
            document.getElementById('param-fuel').textContent = `${pace.fuel_burn_effect_sec_per_lap} s/lap`;
        }
        if (pit) {
            document.getElementById('param-pit-green').textContent = `${pit.green_flag_pit_loss_sec} s`;
            document.getElementById('param-pit-vsc').textContent = `${pit.vsc_pit_loss_sec} s`;
            document.getElementById('param-pit-sc').textContent = `${pit.sc_pit_loss_sec} s`;
        }
    }
}

function renderGapChart(chartData) {
    const laps = chartData.laps;
    const traces = [];

    // Colors mapping
    const teamColors = {
        'SAI': '#E8002D',
        'NOR': '#FF8000',
        'HAM': '#27F4D2',
        'LEC': '#FF334B',
        'VER': '#3671C6',
        'GAS': '#FF87BC',
        'PIA': '#FFB066',
        'PER': '#64C4FF',
        'LAW': '#5E8FAA',
        'MAG': '#B6BABD'
    };

    // Plot drivers (focus on top 5 for visual clarity, plus modified driver)
    const driversToPlot = ['SAI', 'NOR', 'HAM', 'LEC', 'VER'];
    const modCode = officialClassification.find(d => d.driver_number === selectedDriver)?.code;
    if (modCode && !driversToPlot.includes(modCode)) {
        driversToPlot.push(modCode);
    }

    driversToPlot.forEach(code => {
        const d = chartData.drivers[code];
        if (!d) return;

        const color = teamColors[code] || '#FFFFFF';
        const isMod = d.is_modified;

        // Simulated line (solid)
        traces.push({
            x: laps,
            y: d.simulated_gap,
            mode: 'lines',
            name: `${code} (Simulated)`,
            line: {
                color: isMod ? '#F59E0B' : color,
                width: isMod ? 3.5 : 2.2,
                dash: 'solid'
            }
        });

        // Actual line (dashed)
        traces.push({
            x: laps,
            y: d.actual_gap,
            mode: 'lines',
            name: `${code} (Actual)`,
            visible: isMod ? true : 'legendonly',
            line: {
                color: isMod ? '#D97706' : color,
                width: 1.5,
                dash: 'dot'
            }
        });
    });

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'rgba(14, 16, 21, 0.7)',
        font: { color: '#9ca3af', family: 'system-ui, sans-serif', size: 11 },
        margin: { l: 45, r: 20, t: 20, b: 40 },
        xaxis: {
            title: 'Race Lap',
            gridcolor: '#2a2f3d',
            zerolinecolor: '#383d4e',
            tickmode: 'linear',
            dtick: 10
        },
        yaxis: {
            title: 'Gap to Leader (s)',
            gridcolor: '#2a2f3d',
            zerolinecolor: '#383d4e',
            autorange: 'reversed' // P1 at top
        },
        legend: {
            orientation: 'h',
            y: -0.22,
            x: 0,
            font: { size: 10 }
        },
        shapes: [
            // SC shaded band (L20 to L22)
            {
                type: 'rect',
                xref: 'x',
                yref: 'paper',
                x0: 20,
                x1: 22,
                y0: 0,
                y1: 1,
                fillcolor: 'rgba(245, 158, 11, 0.15)',
                line: { width: 0 }
            },
            // VSC shaded band (L44 to L45)
            {
                type: 'rect',
                xref: 'x',
                yref: 'paper',
                x0: 44,
                x1: 45,
                y0: 0,
                y1: 1,
                fillcolor: 'rgba(168, 85, 247, 0.15)',
                line: { width: 0 }
            }
        ]
    };

    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot('gap-chart', traces, layout, config);
}

// Validation Tabs Logic
function switchValTab(tab) {
    ['replay', 'backtest', 'sanity'].forEach(t => {
        const panel = document.getElementById(`val-panel-${t}`);
        const btn = document.getElementById(`tab-btn-${t}`);
        if (t === tab) {
            panel.classList.remove('hidden');
            btn.classList.add('active');
            btn.classList.remove('text-gray-400');
        } else {
            panel.classList.add('hidden');
            btn.classList.remove('active');
            btn.classList.add('text-gray-400');
        }
    });
}

async function loadValidationData() {
    try {
        // 1. Replay Check
        const repRes = await fetch('/api/validation/replay');
        const repData = await repRes.json();
        renderReplayTab(repData);

        // 2. Model Backtest
        const bckRes = await fetch('/api/validation/backtest');
        const bckData = await bckRes.json();
        renderBacktestTab(bckData);

        // 3. Sanity Checks
        const sanRes = await fetch('/api/validation/sanity');
        const sanData = await sanRes.json();
        renderSanityTab(sanData);
    } catch (err) {
        console.error('Failed to load validation checks:', err);
    }
}

function renderReplayTab(data) {
    const tbody = document.getElementById('replay-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    data.table.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="py-2 px-2 font-bold text-white">P${r.replayed_position}</td>
            <td class="py-2 px-2 text-gray-400">P${r.official_position}</td>
            <td class="py-2 px-2">${r.matches ? '<span class="text-emerald-400 font-bold">✓ EXACT</span>' : '<span class="text-rose-500">✗ DIFF</span>'}</td>
            <td class="py-2 px-2 font-bold text-white">${r.code} (${r.name})</td>
            <td class="py-2 px-2 text-gray-300 text-[11px]">${r.team}</td>
            <td class="py-2 px-2 text-gray-400">${r.laps}</td>
            <td class="py-2 px-2 font-bold ${r.replayed_position === 1 ? 'text-emerald-400' : 'text-gray-200'}">${r.replayed_gap}</td>
            <td class="py-2 px-2 text-gray-400 text-[11px]">${r.official_gap}</td>
        `;
        tbody.appendChild(tr);
    });

    const noteEl = document.getElementById('replay-penalty-note');
    if (noteEl) {
        noteEl.innerHTML = `
            <strong class="font-bold text-white block mb-1">Time Penalty Reconciliation Note:</strong>
            <p>${data.penalty_reconciliation_note}</p>
        `;
    }
}

function renderBacktestTab(data) {
    const mMAE = document.getElementById('backtest-model-mae');
    if (!mMAE) return;
    mMAE.textContent = `${data.model_mae_sec.toFixed(3)} s/lap`;
    const bMAE = document.getElementById('backtest-base-mae');
    if (bMAE) bMAE.textContent = `${data.baseline_mae_sec.toFixed(3)} s/lap`;
    const bDelta = document.getElementById('backtest-delta');
    if (bDelta) bDelta.textContent = `${data.delta_mae_sec > 0 ? '+' : ''}${data.delta_mae_sec.toFixed(3)} s/lap`;

    const statusTag = document.getElementById('backtest-status-tag');
    if (statusTag) {
        if (data.beats_baseline) {
            statusTag.textContent = '✓ Model Beats Baseline';
            statusTag.className = 'text-[11px] text-emerald-400 block mt-1 font-bold';
        } else {
            statusTag.textContent = 'Baseline Comparable (Honest Limitation Noted)';
            statusTag.className = 'text-[11px] text-amber-400 block mt-1';
        }
    }

    const compBody = document.getElementById('backtest-compound-body');
    if (compBody) {
        compBody.innerHTML = '';
        data.compound_breakdown.forEach(c => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="py-2 px-2 font-bold text-white">${c.compound}</td>
                <td class="py-2 px-2 text-gray-400">${c.test_laps}</td>
                <td class="py-2 px-2 font-bold text-emerald-400">${c.model_mae.toFixed(3)} s</td>
                <td class="py-2 px-2 text-gray-300">${c.baseline_mae.toFixed(3)} s</td>
                <td class="py-2 px-2">${c.model_beats_baseline ? '<span class="text-emerald-400">✓ Model Beats Baseline</span>' : '<span class="text-gray-400">Baseline Closer</span>'}</td>
            `;
            compBody.appendChild(tr);
        });
    }
}

function renderSanityTab(data) {
    const container = document.getElementById('sanity-checks-container');
    if (!container) return;
    container.innerHTML = '';

    data.checks.forEach(c => {
        const div = document.createElement('div');
        div.className = 'bg-gray-900/60 p-3 rounded-lg border border-gray-800 flex items-start gap-3 text-xs';
        div.innerHTML = `
            <span class="${c.status === 'PASS' ? 'text-emerald-400' : 'text-rose-400'} font-bold text-base mt-0.5">
                ${c.status === 'PASS' ? '✓' : '✗'}
            </span>
            <div>
                <strong class="font-bold text-white">${c.name} [${c.status}]</strong>
                <p class="text-gray-400 mt-0.5">${c.detail}</p>
            </div>
        `;
        container.appendChild(div);
    });

    const fbContainer = document.getElementById('fallbacks-log-container');
    if (fbContainer) {
        fbContainer.innerHTML = '';
        if (data.fallbacks_active && data.fallbacks_active.length > 0) {
            data.fallbacks_active.forEach(fb => {
                const p = document.createElement('p');
                p.innerHTML = `<span class="text-amber-400 font-bold">• ${fb.field}:</span> ${fb.status} — <em>${fb.fallback_applied}</em>`;
                fbContainer.appendChild(p);
            });
        } else {
            fbContainer.innerHTML = '<p class="text-gray-400">No active fallbacks required for loaded telemetry.</p>';
        }
    }
}
