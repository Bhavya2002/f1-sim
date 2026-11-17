/**
 * Step-by-Step Interactive Tutorial Walkthrough for F1 Simulator.
 */

const tutorialSteps = [
    {
        targetId: 'header-banner',
        title: '🏁 Welcome to the F1 Strategy What-If Simulator',
        content: 'This platform starts from the real historical 2023 Singapore Grand Prix (Carlos Sainz win, Lap 20 Safety Car, Lap 44 VSC, Mercedes medium tyre chase). You can change tactical decisions, re-simulate lap-by-lap on a unified arithmetic engine, and compare alternate outcomes.',
        position: 'bottom'
    },
    {
        targetId: 'quick-presets-section',
        title: '⚡ Quick-Start Scenario Presets',
        content: 'Explore classic what-ifs with a single click: replay actual recorded decisions, cancel the Lap 20 Safety Car or Lap 44 VSC, keep Mercedes on a 1-stop, or test a late-race Norris charge on Softs.',
        position: 'bottom'
    },
    {
        targetId: 'ai-chat-section',
        title: '🤖 Natural Language Strategy Assistant',
        content: 'You can give prompts or ask questions in natural language! Try commands like <em>"Make Sainz pit one lap earlier"</em>, <em>"What if there was no safety car?"</em>, or ask <em>"Why did Verstappen win without SC?"</em>. Powered by a local offline NLP intent model.',
        position: 'bottom'
    },
    {
        targetId: 'strategy-studio-section',
        title: '🛠️ What-If Strategy Studio',
        content: 'Select any driver from the top 10. Switch between 1-stop, 2-stop, and 3-stop strategies. Adjust pit stop laps with interactive sliders and change tyre compounds (Soft, Medium, Hard). You can also click <em>🔬 ML Lab ↗</em> to inspect driver telemetry directly.',
        position: 'right'
    },
    {
        targetId: 'environment-controls-section',
        title: '⚙️ Race Interventions & ML Physics Models',
        content: 'Toggle two trained machine learning models: <strong>Model 1 (Traffic & Dirty Air Dynamics)</strong> which calculates overtake probabilities and dirty air penalties (+0.75s), and <strong>Model 2 (Non-Linear Tyre Cliff)</strong> simulating thermal cliffs.',
        position: 'right'
    },
    {
        targetId: 'finishing-order-section',
        title: '📋 Side-by-Side Finishing Order',
        content: 'Compare actual vs. simulated P1–P10 finishing positions. View position deltas (▲/▼), final gaps to the leader, and real-time ML dirty air battles. Clicking any dirty air badge opens that battle in the ML Lab.',
        position: 'left'
    },
    {
        targetId: 'gap-chart-section',
        title: '📈 Lap-by-Lap Gap to Leader Telemetry',
        content: 'High-performance interactive Plotly chart showing gap to the leader over all 62 laps. Solid lines show simulated trajectories, dashed lines show real recorded laps, with amber and purple bands marking Safety Car and VSC periods.',
        position: 'left'
    },
    {
        targetId: 'header-banner',
        title: '🧠 Multi-Page Deep Intelligence',
        content: 'Use the top navigation to explore the <strong>ML Lab</strong> (test overtakes, tyre degradation curves, and NLP inference) and the <strong>Validation Center</strong> (verify the 10/10 replay proof and held-out backtest metrics).',
        position: 'bottom'
    }
];

let currentStepIdx = 0;
let tourOverlay = null;
let tourCard = null;

function startTutorial() {
    currentStepIdx = 0;
    createTourElements();
    showStep(currentStepIdx);
}

function createTourElements() {
    if (document.getElementById('tour-overlay')) return;

    // Overlay backdrop
    tourOverlay = document.createElement('div');
    tourOverlay.id = 'tour-overlay';
    tourOverlay.className = 'fixed inset-0 bg-black/75 z-50 transition-opacity duration-300 pointer-events-auto';
    document.body.appendChild(tourOverlay);

    // Tour Dialog Card
    tourCard = document.createElement('div');
    tourCard.id = 'tour-card';
    tourCard.className = 'fixed z-50 bg-f1card border-2 border-f1red rounded-xl p-5 shadow-2xl max-w-md w-full transition-all duration-300 space-y-3';
    document.body.appendChild(tourCard);

    tourOverlay.onclick = endTutorial;
}

function showStep(idx) {
    if (idx < 0 || idx >= tutorialSteps.length) {
        endTutorial();
        return;
    }

    const step = tutorialSteps[idx];
    const targetEl = document.getElementById(step.targetId);

    // Remove spotlight from previous elements
    document.querySelectorAll('.tour-spotlight').forEach(el => el.classList.remove('tour-spotlight'));

    if (targetEl) {
        targetEl.classList.add('tour-spotlight');
        targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // Position Tour Card
    setTimeout(() => {
        positionTourCard(targetEl, step.position);
    }, 250);

    const isFirst = idx === 0;
    const isLast = idx === tutorialSteps.length - 1;

    tourCard.innerHTML = `
        <div class="flex items-center justify-between border-b border-f1border pb-2.5">
            <span class="text-xs uppercase tracking-wider font-bold text-f1red flex items-center gap-1.5">
                <span>Step ${idx + 1} of ${tutorialSteps.length}</span>
            </span>
            <button onclick="endTutorial()" class="text-gray-400 hover:text-white text-xs font-bold px-1.5 py-0.5 rounded hover:bg-gray-800">
                ✕ Skip Tour
            </button>
        </div>
        <h3 class="text-sm font-bold text-white tracking-tight">${step.title}</h3>
        <p class="text-xs text-gray-300 leading-relaxed">${step.content}</p>
        <div class="flex items-center justify-between pt-2 border-t border-f1border/60">
            <div class="flex gap-1">
                ${tutorialSteps.map((_, i) => `<span class="w-1.5 h-1.5 rounded-full ${i === idx ? 'bg-f1red' : 'bg-gray-700'}"></span>`).join('')}
            </div>
            <div class="flex items-center gap-2">
                ${!isFirst ? `<button onclick="prevStep()" class="text-xs px-2.5 py-1 text-gray-300 hover:text-white bg-gray-800 rounded font-semibold">Previous</button>` : ''}
                <button onclick="${isLast ? 'endTutorial()' : 'nextStep()'}" class="text-xs px-3.5 py-1 bg-f1red hover:bg-red-700 text-white font-bold rounded shadow-md shadow-f1red/20">
                    ${isLast ? 'Finish Tour 🏁' : 'Next →'}
                </button>
            </div>
        </div>
    `;
}

function positionTourCard(targetEl, preferredPos) {
    if (!tourCard) return;

    const cardWidth = Math.min(440, window.innerWidth - 40);
    const cardHeight = 240;

    if (!targetEl) {
        // Fallback: center on screen
        tourCard.style.top = `${Math.max(40, (window.innerHeight - cardHeight) / 2)}px`;
        tourCard.style.left = `${Math.max(20, (window.innerWidth - cardWidth) / 2)}px`;
        return;
    }

    const rect = targetEl.getBoundingClientRect();
    let top = rect.bottom + 15;
    let left = rect.left + (rect.width / 2) - (cardWidth / 2);

    // Keep within viewport bounds
    if (left < 20) left = 20;
    if (left + cardWidth > window.innerWidth - 20) {
        left = window.innerWidth - cardWidth - 20;
    }
    if (top + cardHeight > window.innerHeight - 20) {
        top = Math.max(20, rect.top - cardHeight - 15);
    }

    tourCard.style.top = `${top}px`;
    tourCard.style.left = `${left}px`;
}

function nextStep() {
    currentStepIdx++;
    showStep(currentStepIdx);
}

function prevStep() {
    currentStepIdx--;
    showStep(currentStepIdx);
}

function endTutorial() {
    document.querySelectorAll('.tour-spotlight').forEach(el => el.classList.remove('tour-spotlight'));
    if (tourOverlay) tourOverlay.remove();
    if (tourCard) tourCard.remove();
    tourOverlay = null;
    tourCard = null;
}
