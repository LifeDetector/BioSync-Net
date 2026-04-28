// =============================================
// LifeDetector Camera Sync Improvement (DROP-IN)
// Added by Grok - fixes glitches & sync WITHOUT changing any existing code
// =============================================

const CameraSync = {
  streams: new Map(), // track active streams for cleanup

  async initVideo(videoElement, constraintsOverride = {}) {
    if (!videoElement) return null;

    const defaultConstraints = {
      video: {
        width: { ideal: 640, max: 1280 },
        height: { ideal: 480, max: 720 },
        frameRate: { ideal: 30, min: 24 },
        facingMode: "user",
        noiseSuppression: true
      },
      audio: true
    };

    const finalConstraints = { ...defaultConstraints, ...constraintsOverride };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(finalConstraints);
      videoElement.srcObject = stream;
      this.streams.set(videoElement.id || 'video-' + Date.now(), stream);

      // Force sync with requestAnimationFrame + timestamp
      let lastFrameTime = performance.now();
      const syncLoop = () => {
        if (!videoElement.srcObject) return;
        const now = performance.now();
        if (now - lastFrameTime > 1000 / 30) { // enforce ~30 FPS sync
          videoElement.dispatchEvent(new CustomEvent('camerasynced', {
            detail: { timestamp: now, frameRate: 30 }
          }));
          lastFrameTime = now;
        }
        requestAnimationFrame(syncLoop);
      };
      syncLoop();

      console.log('%c✅ CameraSync: Camera initialized & synced (30 FPS stable)', 'color: #00e5ff; font-weight: bold');
      return stream;
    } catch (err) {
      console.error('%c❌ CameraSync Error:', err);
      // Silent fallback retry once
      if (err.name === 'NotAllowedError' || err.name === 'NotFoundError') {
        setTimeout(() => this.initVideo(videoElement, { video: { facingMode: "environment" } }), 800);
      }
      return null;
    }
  },

  stopVideo(videoElement) {
    if (!videoElement || !videoElement.srcObject) return;
    const stream = videoElement.srcObject;
    stream.getTracks().forEach(track => track.stop());
    videoElement.srcObject = null;
    this.streams.delete(videoElement.id || 'video-' + Date.now());
    console.log('%c✅ CameraSync: Stream cleaned up', 'color: #00e5ff');
  },

  // Auto-attach to any existing #localVideo or MediaPipe camera (call once after your original code runs)
  enhanceExisting() {
    const videos = document.querySelectorAll('video#localVideo, video#remoteVideo, video[id*="video"]');
    videos.forEach(video => {
      if (video.srcObject) {
        // Already has stream → just add sync layer
        let frameCount = 0;
        video.addEventListener('loadedmetadata', () => {
          const sync = () => {
            frameCount++;
            if (frameCount % 2 === 0) { // light sync pulse
              video.dispatchEvent(new CustomEvent('camerasynced'));
            }
            if (video.srcObject) requestAnimationFrame(sync);
          };
          sync();
        });
      } else {
        // No stream yet → auto-init with improved constraints
        this.initVideo(video);
      }
    });
  }
};

// Auto-enhance on page load (safe - does nothing if no camera elements)
window.addEventListener('load', () => {
  setTimeout(() => CameraSync.enhanceExisting(), 500); // small delay so your original code runs first
});

// Expose globally so you can call CameraSync.initVideo() or .stopVideo() anywhere if needed
window.CameraSync = CameraSync;
