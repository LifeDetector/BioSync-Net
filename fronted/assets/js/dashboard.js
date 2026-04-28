// State Signals
const livenessScore = new Signal(0);
const isScanning = new Signal(false);

// DOM Elements
const elements = {
    scoreValue: document.getElementById('livenessScore'),
    btnStart: document.getElementById('btnStartAnalysis'),
    btnLive: document.getElementById('btnLiveCamera'),
    btnExport: document.getElementById('btnExportReport'),
    videoZone: document.getElementById('videoDropZone'),
    placeholder: document.querySelector('.placeholder-text'),
    video: document.getElementById('mainVideo'),
    laser: document.getElementById('scanLaser'),
    visualizer: document.getElementById('lifeVisualizer'),
    logWindow: document.getElementById('terminalLogWindow'),
    syncDriftValue: document.getElementById('syncDriftValue')
};

// --- Terminal Logger ---
function logMsg(msg, type = 'info') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    const timestamp = new Date().toISOString().split('T')[1].slice(0, 8);
    let prefix = '[INFO]';
    if (type === 'success') prefix = '[SUCCESS]';
    if (type === 'warning') prefix = '[WARNING]';
    if (type === 'error') prefix = '[ERROR]';
    
    entry.textContent = `${timestamp} ${prefix} ${msg}`;
    elements.logWindow.appendChild(entry);
    elements.logWindow.scrollTop = elements.logWindow.scrollHeight;
}

// --- Bindings ---
livenessScore.subscribe(score => {
    elements.scoreValue.textContent = score;
    if (score > 80) {
        elements.scoreValue.style.color = 'var(--primary-neon)';
    } else if (score > 40) {
        elements.scoreValue.style.color = 'var(--warning-amber)';
    } else {
        elements.scoreValue.style.color = 'var(--danger-red)';
    }
});

isScanning.subscribe(scanning => {
    if (scanning) {
        elements.laser.classList.add('active');
        elements.visualizer.classList.add('active');
        elements.placeholder.style.display = 'none';
        elements.video.style.display = 'block';
        elements.video.play().catch(e => console.log('Autoplay prevented'));
        logMsg('Initiating multi-modal deep forensic scan...', 'info');
    } else {
        elements.laser.classList.remove('active');
        elements.visualizer.classList.remove('active');
        elements.video.pause();
    }
});

// --- Actions ---
elements.btnStart.addEventListener('click', () => {
    if (isScanning.value) return;
    
    isScanning.value = true;
    livenessScore.value = 0;
    elements.syncDriftValue.textContent = "0";
    logMsg('Extracting rPPG (photoplethysmography) signals...', 'info');
    
    // Simulate Score Counting Up
    let currentScore = 0;
    const targetScore = Math.floor(Math.random() * 20) + 75; // 75-95%
    
    const scoreInterval = setInterval(() => {
        currentScore += Math.floor(Math.random() * 5) + 1;
        if (currentScore >= targetScore) {
            currentScore = targetScore;
            clearInterval(scoreInterval);
            logMsg(`Vitality verified. Final Liveness Score: ${currentScore}%`, 'success');
            isScanning.value = false;
        }
        livenessScore.value = currentScore;
        elements.syncDriftValue.textContent = (Math.random() * 12).toFixed(1);
    }, 100);

    // Populate Charts with Mock Data Animations
    startChartAnimations();
});

elements.btnLive.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        elements.video.srcObject = stream;
        elements.placeholder.style.display = 'none';
        elements.video.style.display = 'block';
        elements.video.play();
        logMsg('Live camera feed established.', 'success');
        elements.video.removeAttribute('src'); 
    } catch (err) {
        logMsg('Failed to access system camera: ' + err.message, 'error');
    }
});

elements.btnExport.addEventListener('click', () => {
    logMsg('Generating formal PDF forensic report...', 'info');
    setTimeout(() => {
        logMsg('Report Exported Successfully. [evidence_report_2026.pdf]', 'success');
    }, 1500);
});

// --- Chart Visualizations (Mock Data) ---
let rppgChart, audioChart, emotionChart;

function initCharts() {
    // rPPG Line Chart
    const ctxRppg = document.getElementById('rppgChart').getContext('2d');
    rppgChart = new Chart(ctxRppg, {
        type: 'line',
        data: {
            labels: Array.from({length: 20}, (_, i) => i),
            datasets: [{
                label: 'Pulse Wave',
                data: Array(20).fill(0),
                borderColor: '#39ff14',
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

    // Audio Spectral Gaps Bar Chart
    const ctxAudio = document.getElementById('audioSpectrumChart').getContext('2d');
    audioChart = new Chart(ctxAudio, {
        type: 'bar',
        data: {
            labels: Array.from({length: 15}, (_, i) => i),
            datasets: [{
                label: 'Frequency',
                data: Array(15).fill(2),
                backgroundColor: '#39ff14'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { display: false, min: 0, max: 100 }
            }
        }
    });

    // Emotion Consistency Radar Chart
    const ctxEmotion = document.getElementById('emotionRadarChart').getContext('2d');
    emotionChart = new Chart(ctxEmotion, {
        type: 'radar',
        data: {
            labels: ['Joy', 'Anger', 'Sadness', 'Surprise', 'Fear', 'Disgust'],
            datasets: [
                {
                    label: 'Voice Emotion',
                    data: [80, 20, 10, 50, 10, 5],
                    borderColor: '#39ff14',
                    backgroundColor: 'rgba(57, 255, 20, 0.2)',
                    borderWidth: 1,
                    pointRadius: 0
                },
                {
                    label: 'Facial Emotion',
                    data: [75, 25, 15, 45, 15, 10],
                    borderColor: '#8e8e93',
                    backgroundColor: 'rgba(142, 142, 147, 0.2)',
                    borderWidth: 1,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                r: {
                    angleLines: { color: '#1a1a1a' },
                    grid: { color: '#1a1a1a' },
                    pointLabels: { color: '#8e8e93', font: { size: 10 } },
                    ticks: { display: false }
                }
            }
        }
    });
}

function startChartAnimations() {
    let tick = 0;
    const animInterval = setInterval(() => {
        if (!isScanning.value) {
            clearInterval(animInterval);
            return;
        }
        
        // Update rPPG
        const rppgData = rppgChart.data.datasets[0].data;
        rppgData.shift();
        rppgData.push(Math.sin(tick * 0.5) * 5 + (Math.random() * 2 - 1));
        rppgChart.update();

        // Update Audio
        const audioData = audioChart.data.datasets[0].data;
        for (let i = 0; i < audioData.length; i++) {
            audioData[i] = Math.random() * 100;
        }
        audioChart.update();

        // Update Radar slightly
        const faceData = emotionChart.data.datasets[1].data;
        for (let i = 0; i < faceData.length; i++) {
            faceData[i] = faceData[i] + (Math.random() * 4 - 2);
        }
        emotionChart.update();

        tick++;
    }, 100);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
});
