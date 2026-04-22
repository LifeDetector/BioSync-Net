// ─── State Signals ────────────────────────────────────────────────────────────
const isEngineOpen = new Signal(false);
const isScanning = new Signal(false);
const syncScore = new Signal(100);

// ─── DOM Elements ─────────────────────────────────────────────────────────────
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

// ─── Panel Toggle ─────────────────────────────────────────────────────────────
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

// ─── Scan Logic ───────────────────────────────────────────────────────────────
elements.btnStartScan.addEventListener('click', () => {
    if (isScanning.value) return;
    isScanning.value = true;
    syncScore.value = 99;

    elements.btnStartScan.textContent = 'SCANNING IN PROGRESS...';
    elements.btnStartScan.classList.add('scanning');
    elements.miniCharts.classList.add('active');
    elements.overlay.classList.remove('hidden');
    elements.overlay.classList.remove('danger');
    elements.toast.classList.add('hidden');

    startAnalysisSimulation();
});

// ─── Simulation Logic ─────────────────────────────────────────────────────────
let rppgChartMeet;

function initMeetCharts() {
    const ctx = document.getElementById('meetRppgChart').getContext('2d');
    rppgChartMeet = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: 30 }, (_, i) => i),
            datasets: [{
                label: 'Pulse',
                data: Array(30).fill(0),
                borderColor: '#1a73e8',
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
        const rppgData = rppgChartMeet.data.datasets[0].data;
        rppgData.shift();

        let noise = Math.random() * 2 - 1;
        if (tick > 50) {
            noise = Math.random() * 8 - 4;
            rppgChartMeet.data.datasets[0].borderColor = '#ea4335';
        }

        rppgData.push(Math.sin(tick * 0.4) * 4 + noise);
        rppgChartMeet.update();

        if (tick > 20 && syncScore.value > 35) {
            let drop = tick > 40
                ? Math.floor(Math.random() * 5) + 2
                : Math.floor(Math.random() * 2);
            syncScore.value = Math.max(32, syncScore.value - drop);
        }

        tick++;

        if (tick >= 80) {
            clearInterval(simInterval);
            elements.btnStartScan.textContent = 'SYNTHETIC DETECTED';
            elements.btnStartScan.style.backgroundColor = '#ea4335';
            elements.btnStartScan.style.color = 'white';
        }
    }, 100);
}

// ─── Score Bindings ───────────────────────────────────────────────────────────
syncScore.subscribe(score => {
    elements.syncScoreDisplay.textContent = score;
    elements.safetyMeterFill.style.width = `${score}%`;

    if (score < 40) {
        elements.safetyMeterFill.className = 'meter-fill red';
        elements.syncScoreDisplay.style.color = '#ea4335';
        if (isScanning.value) {
            elements.overlay.classList.add('danger');
            elements.toast.classList.remove('hidden');
        }
    } else {
        elements.safetyMeterFill.className = 'meter-fill green';
        elements.syncScoreDisplay.style.color = '#202124';
    }
});

// ─── Meet UI Controls ─────────────────────────────────────────────────────────
function updateClock() {
    const now = new Date();
    let hours = now.getHours();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12 || 12;
    const minutes = now.getMinutes().toString().padStart(2, '0');
    elements.timeDisplay.textContent = `${hours}:${minutes} ${ampm} | qwe-rtzy-uip`;
}
setInterval(updateClock, 1000);
updateClock();

elements.btnEndCall.addEventListener('click', () => {
    if (ws) ws.close();
    if (peerConnection) peerConnection.close();
    window.location.href = '../index.html';
});

// ─── Local Camera ─────────────────────────────────────────────────────────────
let localStream = null;

// ─── WebRTC + WebSocket Signaling ─────────────────────────────────────────────
const roomId = new URLSearchParams(window.location.search).get('room') || 'default-room';
const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
const wsUrl = `${wsProtocol}://${window.location.host}/ws/${roomId}`;

let ws = null;
let peerConnection = null;
let isInitiator = false;

const iceConfig = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' }
    ]
};

// Get or create the remote video element
function getRemoteVideo() {
    let remoteVideo = document.getElementById('remoteVideo');
    if (!remoteVideo) {
        // Create a picture-in-picture style remote video if it doesn't exist
        remoteVideo = document.createElement('video');
        remoteVideo.id = 'remoteVideo';
        remoteVideo.autoplay = true;
        remoteVideo.playsinline = true;
        remoteVideo.style.cssText = `
            position: absolute;
            bottom: 16px;
            right: 16px;
            width: 200px;
            height: 150px;
            border-radius: 8px;
            object-fit: cover;
            border: 2px solid #1a73e8;
            background: #000;
            z-index: 10;
        `;
        const stage = document.querySelector('.main-stage') || document.body;
        stage.appendChild(remoteVideo);
    }
    return remoteVideo;
}

// Show a connection status badge
function setStatus(msg, color = '#1a73e8') {
    let badge = document.getElementById('rtcStatusBadge');
    if (!badge) {
        badge = document.createElement('div');
        badge.id = 'rtcStatusBadge';
        badge.style.cssText = `
            position: fixed;
            top: 12px;
            left: 50%;
            transform: translateX(-50%);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            color: #fff;
            z-index: 9999;
            transition: background 0.3s;
        `;
        document.body.appendChild(badge);
    }
    badge.style.background = color;
    badge.textContent = msg;
}

async function createPeerConnection() {
    if (peerConnection) {
        peerConnection.close();
    }

    peerConnection = new RTCPeerConnection(iceConfig);

    // Add local tracks so the remote peer can see/hear us
    if (localStream) {
        localStream.getTracks().forEach(track => {
            peerConnection.addTrack(track, localStream);
        });
    }

    // When remote media arrives, show it
    peerConnection.ontrack = event => {
        const remoteVideo = getRemoteVideo();
        if (remoteVideo.srcObject !== event.streams[0]) {
            remoteVideo.srcObject = event.streams[0];
            setStatus('🟢 Connected to peer', '#0f9d58');
        }
    };

    // Relay our ICE candidates through the server
    peerConnection.onicecandidate = event => {
        if (event.candidate && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'ice',
                candidate: event.candidate,
                room: roomId
            }));
        }
    };

    peerConnection.onconnectionstatechange = () => {
        const state = peerConnection.connectionState;
        if (state === 'connected') {
            setStatus('🟢 Connected to peer', '#0f9d58');
        } else if (state === 'disconnected' || state === 'failed') {
            setStatus('🔴 Peer disconnected', '#ea4335');
        } else if (state === 'connecting') {
            setStatus('🔵 Connecting...', '#1a73e8');
        }
    };

    return peerConnection;
}

async function startCall() {
    await createPeerConnection();
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    ws.send(JSON.stringify({ type: 'offer', sdp: offer, room: roomId }));
    setStatus('📡 Sending offer...', '#f4b400');
}

function initWebSocket() {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('[WS] Connected to signaling server, room:', roomId);
        setStatus('⏳ Waiting for peer...', '#f4b400');
    };

    ws.onmessage = async event => {
        let msg;
        try {
            msg = JSON.parse(event.data);
        } catch {
            return;
        }

        switch (msg.type) {
            case 'room_info':
                // Server tells us how many peers are in the room
                if (msg.peer_count >= 2) {
                    // We are the second peer (subject) — redirect to subject dashboard
                    window.location.href = `/dashboard/subject.html?room=${roomId}`;
                }
                break;

            case 'ready':
                // Another peer joined — we (the first peer) initiate the call
                setStatus('🔵 Peer joined, starting call...', '#1a73e8');
                await startCall();
                break;

            case 'offer':
                await createPeerConnection();
                await peerConnection.setRemoteDescription(
                    new RTCSessionDescription(msg.sdp)
                );
                const answer = await peerConnection.createAnswer();
                await peerConnection.setLocalDescription(answer);
                ws.send(JSON.stringify({ type: 'answer', sdp: answer, room: roomId }));
                setStatus('📡 Answering call...', '#f4b400');
                break;

            case 'answer':
                if (peerConnection) {
                    await peerConnection.setRemoteDescription(
                        new RTCSessionDescription(msg.sdp)
                    );
                }
                break;

            case 'ice':
                if (peerConnection && msg.candidate) {
                    try {
                        await peerConnection.addIceCandidate(
                            new RTCIceCandidate(msg.candidate)
                        );
                    } catch (e) {
                        console.warn('[ICE] Failed to add candidate:', e);
                    }
                }
                break;

            case 'peer_left':
                setStatus('🔴 Peer left the room', '#ea4335');
                const rv = document.getElementById('remoteVideo');
                if (rv) rv.srcObject = null;
                if (peerConnection) {
                    peerConnection.close();
                    peerConnection = null;
                }
                break;
        }
    };

    ws.onerror = err => {
        console.error('[WS] Error:', err);
        setStatus('⚠️ Connection error', '#ea4335');
    };

    ws.onclose = () => {
        console.log('[WS] Disconnected');
        setStatus('🔴 Disconnected', '#ea4335');
    };
}

// ─── Initialise Everything ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    initMeetCharts();

    const mainVideo = document.getElementById('mainMeetVideo');
    if (mainVideo) {
        try {
            localStream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: true
            });
            mainVideo.srcObject = localStream;

            // Mute toggle
            elements.btnMute.addEventListener('click', () => {
                const audioTrack = localStream.getAudioTracks()[0];
                if (audioTrack) {
                    audioTrack.enabled = !audioTrack.enabled;
                    elements.btnMute.classList.toggle('off', !audioTrack.enabled);
                }
            });

            // Video toggle
            elements.btnVideo.addEventListener('click', () => {
                const videoTrack = localStream.getVideoTracks()[0];
                if (videoTrack) {
                    videoTrack.enabled = !videoTrack.enabled;
                    elements.btnVideo.classList.toggle('off', !videoTrack.enabled);
                }
            });

            // Screen share
            elements.btnShare.addEventListener('click', async () => {
                try {
                    const displayStream = await navigator.mediaDevices.getDisplayMedia({
                        video: true
                    });
                    mainVideo.srcObject = displayStream;
                    elements.btnShare.classList.add('active');

                    // If in a call, replace the video track sent to peer
                    if (peerConnection) {
                        const videoSender = peerConnection
                            .getSenders()
                            .find(s => s.track && s.track.kind === 'video');
                        if (videoSender) {
                            videoSender.replaceTrack(displayStream.getVideoTracks()[0]);
                        }
                    }

                    displayStream.getVideoTracks()[0].onended = () => {
                        mainVideo.srcObject = localStream;
                        elements.btnShare.classList.remove('active');
                        // Restore camera track to peer
                        if (peerConnection) {
                            const videoSender = peerConnection
                                .getSenders()
                                .find(s => s.track && s.track.kind === 'video');
                            if (videoSender) {
                                videoSender.replaceTrack(localStream.getVideoTracks()[0]);
                            }
                        }
                    };
                } catch (err) {
                    console.error('Screen sharing failed or was cancelled:', err);
                }
            });

        } catch (err) {
            console.error('Failed to access camera/microphone:', err);
            setStatus('⚠️ Camera access denied', '#ea4335');
        }
    }

    // Start WebSocket after camera is ready
    initWebSocket();
});
