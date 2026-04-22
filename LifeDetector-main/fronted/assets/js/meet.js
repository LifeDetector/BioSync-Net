// State Signals
const isEngineOpen = new Signal(false);
const isScanning = new Signal(false);
const syncScore = new Signal(100);

// DOM Elements
const elements = {
    mainStage: document.querySelector('.main-stage'),
    btnToggleEngine: document.getElementById('btnToggleEngine'),
    securityPanel: document.getElementById('securityPanel'),
    btnCloseEngine: document.getElementById('btnCloseEngine'),
    btnStartScan: document.getElementById('btnStartForensicScan'),
    miniCharts: document.querySelector('.mini-charts-container'),
    syncScoreDisplay: document.getElementById('meetSyncScore'),
    safetyMeterFill: document.getElementById('safetyMeterFill'),
    overlay: document.getElementById('detectorOverlay'),
    toast: document.getElementById('criticalAlertToast'),
    btnMute: document.querySelector('.ctrl-btn.mute'),
    btnVideo: document.querySelector('.ctrl-btn.video'),
    btnShare: document.querySelector('.ctrl-btn.share'),
    btnEndCall: document.querySelector('.ctrl-btn.end-call'),
    timeDisplay: document.querySelector('.time-display')
};

// --- Panel Toggle ---
function togglePanel() {
    isEngineOpen.value = !isEngineOpen.value;
}

elements.btnToggleEngine.addEventListener('click', togglePanel);
elements.btnCloseEngine.addEventListener('click', togglePanel);

isEngineOpen.subscribe(isOpen => {
    if (isOpen) {
        elements.securityPanel.classList.add('open');
        elements.mainStage.classList.add('panel-open');
        elements.btnToggleEngine.classList.add('active');
    } else {
        elements.securityPanel.classList.remove('open');
        elements.mainStage.classList.remove('panel-open');
        elements.btnToggleEngine.classList.remove('active');
    }
});

// --- Scan Logic ---
elements.btnStartScan.addEventListener('click', () => {
    if (isScanning.value) return; // Prevent double click
    
    isScanning.value = true;
    syncScore.value = 99; // Start high
    
    elements.btnStartScan.textContent = "SCANNING IN PROGRESS...";
    elements.btnStartScan.classList.add('scanning');
    elements.miniCharts.classList.add('active');
    elements.overlay.classList.remove('hidden');
    elements.overlay.classList.remove('danger');
    elements.toast.classList.add('hidden');
    
    startAnalysisSimulation();
});

// --- Simulation Logic ---
let rppgChartMeet;

function initMeetCharts() {
    const ctx = document.getElementById('meetRppgChart').getContext('2d');
    rppgChartMeet = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 30}, (_, i) => i),
            datasets: [{
                label: 'Pulse',
                data: Array(30).fill(0),
                borderColor: '#1a73e8', // Meet blue for the chart
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { display: false, min: -10, max: 10 }
            }
        }
    });
}

function startAnalysisSimulation() {
    let tick = 0;
    
    const simInterval = setInterval(() => {
        // 1. Update rPPG Chart (Normal sine wave initially, gets erratic later)
        const rppgData = rppgChartMeet.data.datasets[0].data;
        rppgData.shift();
        
        let noise = Math.random() * 2 - 1;
        if (tick > 50) {
            noise = Math.random() * 8 - 4; // Introduce artifact noise
            rppgChartMeet.data.datasets[0].borderColor = '#ea4335'; // Turn red
        }
        
        rppgData.push(Math.sin(tick * 0.4) * 4 + noise);
        rppgChartMeet.update();

        // 2. Drop the Sync Score over time
        if (tick > 20 && syncScore.value > 35) {
            // Drop rapidly after tick 40
            let drop = tick > 40 ? Math.floor(Math.random() * 5) + 2 : Math.floor(Math.random() * 2);
            syncScore.value = Math.max(32, syncScore.value - drop);
        }

        tick++;

        // Stop simulation after 80 ticks
        if (tick >= 80) {
            clearInterval(simInterval);
            elements.btnStartScan.textContent = "SYNTHETIC DETECTED";
            elements.btnStartScan.style.backgroundColor = "#ea4335";
            elements.btnStartScan.style.color = "white";
        }
    }, 100);
}

// --- Bindings ---
syncScore.subscribe(score => {
    elements.syncScoreDisplay.textContent = score;
    
    // Update Safety Meter Width and Color
    elements.safetyMeterFill.style.width = `${score}%`;
    
    if (score < 40) {
        elements.safetyMeterFill.className = 'meter-fill red';
        elements.syncScoreDisplay.style.color = '#ea4335';
        
        // Trigger Critical Alert State
        if (isScanning.value) {
            elements.overlay.classList.add('danger');
            elements.toast.classList.remove('hidden');
        }
    } else {
        elements.safetyMeterFill.className = 'meter-fill green';
        elements.syncScoreDisplay.style.color = '#202124';
    }
});

// --- Meet UI Controls ---
function updateClock() {
    const now = new Date();
    let hours = now.getHours();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12;
    hours = hours ? hours : 12;
    const minutes = now.getMinutes().toString().padStart(2, '0');
    elements.timeDisplay.textContent = `${hours}:${minutes} ${ampm} | qwe-rtzy-uip`;
}

setInterval(updateClock, 1000);
updateClock();

elements.btnEndCall.addEventListener('click', () => {
    window.location.href = '../index.html';
});

let localStream = null;

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    initMeetCharts();
    
    // Initialize Live Camera for Meet Clone
    const mainVideo = document.getElementById('mainMeetVideo');
    if (mainVideo) {
        try {
            localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            mainVideo.srcObject = localStream;
            
            // Wire up mute/video toggles
            elements.btnMute.addEventListener('click', () => {
                const audioTrack = localStream.getAudioTracks()[0];
                if (audioTrack) {
                    audioTrack.enabled = !audioTrack.enabled;
                    elements.btnMute.classList.toggle('off', !audioTrack.enabled);
                }
            });

            elements.btnVideo.addEventListener('click', () => {
                const videoTrack = localStream.getVideoTracks()[0];
                if (videoTrack) {
                    videoTrack.enabled = !videoTrack.enabled;
                    elements.btnVideo.classList.toggle('off', !videoTrack.enabled);
                }
            });

            // Mock Screen Share
            elements.btnShare.addEventListener('click', async () => {
                try {
                    const displayStream = await navigator.mediaDevices.getDisplayMedia({ video: true });
                    mainVideo.srcObject = displayStream;
                    elements.btnShare.classList.add('active'); // highlight share button
                    
                    // Revert when sharing stops
                    displayStream.getVideoTracks()[0].onended = () => {
                        mainVideo.srcObject = localStream;
                        elements.btnShare.classList.remove('active');
                    };
                } catch (err) {
                    console.error('Screen sharing failed or was cancelled:', err);
                }
            });

        } catch (err) {
            console.error('Failed to access system camera:', err);
        }
    }
});
