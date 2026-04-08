/**
 * =============================================================================
 * TABBA Web — Client-Side Application
 * =============================================================================
 * 
 * The Canvas Trick:
 *   Instead of recording video → saving MP4 → extracting frames,
 *   we use HTML5 Canvas to capture frames from the live camera at 3 FPS 
 *   (every 333ms) and send them via WebSocket to the FastAPI server.
 * 
 * Session Protocol:
 *   Phase 1/2 fusion (0-5s): Server accumulates 142 features per frame
 *   Final (t=5s): Server averages features → single XGBoost prediction
 * =============================================================================
 */

// =============================================================================
// STATE
// =============================================================================
const STATE = {
    ws: null,
    stream: null,
    captureInterval: null,
    sessionActive: false,
    sessionStartTime: null,
    framesSent: 0,
    framesReceived: 0,
    fpsCounter: 0,
    fpsTimestamp: null,
    currentFPS: 0,
};

const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const CONFIG = {
    WS_URL: `${protocol}//${window.location.host}/ws`,
    CAPTURE_INTERVAL: 333,       // 3 FPS = 333ms
    JPEG_QUALITY: 0.80,          // JPEG quality for frame capture
    CANVAS_WIDTH: 640,           // Capture resolution
    CANVAS_HEIGHT: 480,
    SESSION_DURATION: 5.0,       // Seconds (strict, non-configurable)
    STABILIZATION_TIME: 0.0,     // Phase 1 -> Phase 2 boundary
};

// =============================================================================
// DOM ELEMENTS
// =============================================================================
const $ = (id) => document.getElementById(id);

const DOM = {
    video: $('camera-video'),
    placeholder: $('camera-placeholder'),
    canvas: $('capture-canvas'),
    statusDot: $('status-dot'),
    statusText: $('status-text'),
    sessionBadge: $('session-badge'),
    resultBadge: $('result-badge'),
    cameraTimer: $('camera-timer'),
    detectionInfo: $('detection-info'),
    bloodStatus: $('blood-status'),
    patchesStatus: $('patches-status'),
    bloodChip: $('blood-chip'),
    patchesChip: $('patches-chip'),
    phaseIndicator: $('phase-indicator'),
    phase1Step: $('phase1-step'),
    phase2Step: $('phase2-step'),
    finalStep: $('final-step'),
    progressSection: $('progress-section'),
    progressFill: $('progress-fill'),
    elapsedLabel: $('elapsed-label'),
    hgWaiting: $('hg-waiting'),
    hgPreview: $('hg-preview'),
    hgPreviewValue: $('hg-preview-value'),
    hgPreviewLabel: $('hg-preview-label'),
    statsRow: $('stats-row'),
    statFrames: $('stat-frames'),
    statPhase2: $('stat-phase2'),
    statFps: $('stat-fps'),
    finalResultCard: $('final-result-card'),
    finalHgValue: $('final-hg-value'),
    finalFrames: $('final-frames'),
    btnStart: $('btn-start'),
    scanLine: $('scan-line'),
};


// =============================================================================
// CAMERA MANAGER
// =============================================================================
async function startCamera() {
    try {
        // Prefer rear camera on mobile
        const constraints = {
            video: {
                facingMode: { ideal: 'environment' },
                width: { ideal: CONFIG.CANVAS_WIDTH },
                height: { ideal: CONFIG.CANVAS_HEIGHT },
            },
            audio: false,
        };

        STATE.stream = await navigator.mediaDevices.getUserMedia(constraints);
        DOM.video.srcObject = STATE.stream;
        await DOM.video.play();

        // Show video, hide placeholder
        DOM.video.classList.remove('hidden');
        DOM.placeholder.classList.add('hidden');

        // Set canvas size
        DOM.canvas.width = CONFIG.CANVAS_WIDTH;
        DOM.canvas.height = CONFIG.CANVAS_HEIGHT;

        updateStatus('active', 'Camera Ready');
        return true;
    } catch (err) {
        console.error('Camera error:', err);
        updateStatus('error', 'Camera Denied');

        DOM.placeholder.querySelector('.camera-placeholder-text').textContent =
            'Camera access denied. Please allow camera permissions.';
        return false;
    }
}

function stopCamera() {
    if (STATE.stream) {
        STATE.stream.getTracks().forEach(track => track.stop());
        STATE.stream = null;
    }
    DOM.video.classList.add('hidden');
    DOM.placeholder.classList.remove('hidden');
    updateStatus('', 'Offline');
}


// =============================================================================
// WEBSOCKET
// =============================================================================
function connectWebSocket() {
    return new Promise((resolve, reject) => {
        if (STATE.ws && STATE.ws.readyState === WebSocket.OPEN) {
            resolve(STATE.ws);
            return;
        }

        STATE.ws = new WebSocket(CONFIG.WS_URL);

        STATE.ws.onopen = () => {
            console.log('[WS] Connected');
            resolve(STATE.ws);
        };

        STATE.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleServerResponse(data);
        };

        STATE.ws.onclose = () => {
            console.log('[WS] Disconnected');
            if (STATE.sessionActive) {
                updateStatus('error', 'Connection Lost');
            }
        };

        STATE.ws.onerror = (err) => {
            console.error('[WS] Error:', err);
            reject(err);
        };

        // Timeout
        setTimeout(() => {
            if (STATE.ws.readyState !== WebSocket.OPEN) {
                reject(new Error('WebSocket connection timeout'));
            }
        }, 5000);
    });
}

function sendMessage(msg) {
    if (STATE.ws && STATE.ws.readyState === WebSocket.OPEN) {
        STATE.ws.send(JSON.stringify(msg));
    }
}


// =============================================================================
// FRAME CAPTURE (The Canvas Trick ✨)
// =============================================================================
function startCapture() {
    const ctx = DOM.canvas.getContext('2d');

    STATE.captureInterval = setInterval(() => {
        if (!STATE.sessionActive) return;

        // Draw current video frame onto canvas
        ctx.drawImage(DOM.video, 0, 0, CONFIG.CANVAS_WIDTH, CONFIG.CANVAS_HEIGHT);

        // Encode as JPEG base64
        const jpegBase64 = DOM.canvas.toDataURL('image/jpeg', CONFIG.JPEG_QUALITY);

        // Calculate elapsed time
        const elapsed = (Date.now() - STATE.sessionStartTime) / 1000;

        // Send to server
        sendMessage({
            action: 'frame',
            frame: jpegBase64,
            time: elapsed.toFixed(2),
        });

        STATE.framesSent++;

        // Update timer
        updateTimer(elapsed);

        // Check if session should end
        if (elapsed >= CONFIG.SESSION_DURATION) {
            // Stop capturing and explicitly ask server for final verdict
            stopCapture();
            sendMessage({ action: 'finalize' });
        }

    }, CONFIG.CAPTURE_INTERVAL);
}

function stopCapture() {
    if (STATE.captureInterval) {
        clearInterval(STATE.captureInterval);
        STATE.captureInterval = null;
    }
}


// =============================================================================
// SERVER RESPONSE HANDLER
// =============================================================================
function handleServerResponse(data) {
    STATE.framesReceived++;

    // FPS calculation
    const now = Date.now();
    STATE.fpsCounter++;
    if (!STATE.fpsTimestamp || now - STATE.fpsTimestamp >= 1000) {
        STATE.currentFPS = STATE.fpsCounter;
        STATE.fpsCounter = 0;
        STATE.fpsTimestamp = now;
    }

    // Update stats
    DOM.statFrames.textContent = data.frame_count || STATE.framesReceived;
    DOM.statPhase2.textContent = data.phase2_frames || 0;
    DOM.statFps.textContent = STATE.currentFPS;

    const status = data.status;
    const phase = data.phase || 1;
    const elapsed = data.elapsed || 0;

    // --- Detection Status ---
    if (data.blood_detected) {
        DOM.bloodStatus.textContent = 'Detected';
        DOM.bloodChip.className = 'detection-chip chip-ok';
    } else if (status === 'no_blood') {
        DOM.bloodStatus.textContent = 'Not Found';
        DOM.bloodChip.className = 'detection-chip chip-error';
    }

    if (data.patches_found >= 8) {
        DOM.patchesStatus.textContent = `${data.patches_found}/8`;
        DOM.patchesChip.className = 'detection-chip chip-ok';
    } else if (data.patches_found > 0) {
        DOM.patchesStatus.textContent = `${data.patches_found}/8`;
        DOM.patchesChip.className = 'detection-chip chip-warn';
    }

    // --- Phase Updates ---
    updatePhaseUI(phase);

    // --- Hg Preview (Phase 1 & 2) ---
    if (data.hg_preview !== null && data.hg_preview !== undefined && !data.is_final) {
        showPreview(data.hg_preview, phase);
    }

    // --- Final Verdict ---
    if (data.is_final && data.hg_final !== null) {
        showFinalResult(data.hg_final, data.frames_used || data.phase2_frames || 0);
    }

    // Update progress
    const pct = Math.min((elapsed / CONFIG.SESSION_DURATION) * 100, 100);
    DOM.progressFill.style.width = `${pct}%`;
    DOM.elapsedLabel.textContent = `${elapsed.toFixed(1)}s`;
}


// =============================================================================
// UI UPDATE FUNCTIONS
// =============================================================================
function updateStatus(type, text) {
    DOM.statusText.textContent = text;
    DOM.statusDot.className = 'status-dot';
    if (type === 'active') DOM.statusDot.classList.add('active');
    if (type === 'recording') DOM.statusDot.classList.add('recording');
}

function updateTimer(elapsed) {
    DOM.cameraTimer.textContent = `${elapsed.toFixed(1)}s`;
}

function updatePhaseUI(phase) {
    // Reset
    DOM.phase1Step.className = 'phase-step';
    DOM.phase2Step.className = 'phase-step';
    DOM.finalStep.className = 'phase-step';

    if (phase >= 1) {
        DOM.phase1Step.classList.add(phase === 1 ? 'active' : 'complete');
    }
    if (phase >= 2) {
        DOM.phase2Step.classList.add(phase === 2 ? 'active' : 'complete');

        // Update session badge
        DOM.sessionBadge.className = 'card-badge badge-phase2';
        DOM.sessionBadge.textContent = 'PHASE 2 · MEASURING';
    }
    if (phase >= 3) {
        DOM.finalStep.classList.add('complete');
    }

    if (phase === 1) {
        DOM.sessionBadge.className = 'card-badge badge-phase1';
        DOM.sessionBadge.textContent = 'PHASE 1 · STABILIZING';
    }
}

function showPreview(hgValue, phase) {
    DOM.hgWaiting.classList.add('hidden');
    DOM.hgPreview.classList.remove('hidden');

    DOM.hgPreviewValue.textContent = hgValue.toFixed(1);
    DOM.hgPreviewValue.className = `hg-value preview ${getHgClass(hgValue)}`;

    if (phase === 1) {
        DOM.hgPreviewLabel.textContent = '⚠️ Live preview — reaction not yet stable';
        DOM.resultBadge.className = 'card-badge badge-phase1';
        DOM.resultBadge.textContent = 'PREVIEW';
    } else if (phase === 2) {
        DOM.hgPreviewLabel.textContent = '📊 Accumulating stable features...';
        DOM.resultBadge.className = 'card-badge badge-phase2';
        DOM.resultBadge.textContent = 'MEASURING';
    }
}

function showFinalResult(hgValue, framesUsed) {
    // Stop everything
    STATE.sessionActive = false;
    stopCapture();

    // Update badges
    DOM.sessionBadge.className = 'card-badge badge-final';
    DOM.sessionBadge.textContent = 'COMPLETE';
    DOM.resultBadge.className = 'card-badge badge-final';
    DOM.resultBadge.textContent = 'FINAL';

    // Update phase UI
    DOM.phase1Step.className = 'phase-step complete';
    DOM.phase2Step.className = 'phase-step complete';
    DOM.finalStep.className = 'phase-step complete';

    // Progress to 100%
    DOM.progressFill.style.width = '100%';
    DOM.progressFill.classList.remove('active');
    DOM.elapsedLabel.textContent = '5.0s';

    // Hide preview, show final
    DOM.hgPreview.classList.add('hidden');
    DOM.statsRow.classList.add('hidden');

    // Final result card
    DOM.finalHgValue.textContent = hgValue.toFixed(1);
    DOM.finalHgValue.className = `final-hg-value ${getHgClass(hgValue)}`;
    DOM.finalFrames.textContent = framesUsed;
    DOM.finalResultCard.classList.add('visible');

    // Scan line off
    DOM.scanLine.classList.remove('active');
    updateStatus('active', 'Complete');

    // Change button to "New Session"
    DOM.btnStart.textContent = '🔄 New Session';
    DOM.btnStart.className = 'btn-primary btn-reset';
    DOM.btnStart.disabled = false;
}

function getHgClass(hg) {
    if (hg < 7) return 'critical-low';
    if (hg < 11) return 'low';
    if (hg <= 17) return 'normal';
    return 'high';
}

function resetUI() {
    // Reset all UI to initial state
    DOM.sessionBadge.className = 'card-badge badge-idle';
    DOM.sessionBadge.textContent = 'IDLE';
    DOM.resultBadge.className = 'card-badge badge-idle';
    DOM.resultBadge.textContent = 'WAITING';

    DOM.cameraTimer.classList.add('hidden');
    DOM.detectionInfo.classList.add('hidden');
    DOM.phaseIndicator.classList.add('hidden');
    DOM.progressSection.classList.add('hidden');
    DOM.scanLine.classList.remove('active');
    DOM.progressFill.style.width = '0%';
    DOM.progressFill.classList.remove('active');

    DOM.hgWaiting.classList.remove('hidden');
    DOM.hgPreview.classList.add('hidden');
    DOM.statsRow.classList.add('hidden');
    DOM.finalResultCard.classList.remove('visible');

    DOM.hgPreviewValue.textContent = '—';
    DOM.bloodStatus.textContent = '—';
    DOM.patchesStatus.textContent = '—';
    DOM.statFrames.textContent = '0';
    DOM.statPhase2.textContent = '0';
    DOM.statFps.textContent = '0';

    DOM.phase1Step.className = 'phase-step';
    DOM.phase2Step.className = 'phase-step';
    DOM.finalStep.className = 'phase-step';

    DOM.btnStart.textContent = '▶ Start Session';
    DOM.btnStart.className = 'btn-primary';
    DOM.btnStart.disabled = false;
}


// =============================================================================
// MAIN CONTROL (Start / Stop / Reset)
// =============================================================================
async function handleStartStop() {
    if (STATE.sessionActive) {
        // Stop session
        endSession();
        return;
    }

    // Check if this is a reset after completion
    const btnText = DOM.btnStart.textContent;
    if (btnText.includes('New Session')) {
        resetUI();
        stopCamera();

        // Give a brief moment for cleanup
        await new Promise(r => setTimeout(r, 300));
    }

    // === START SESSION ===
    DOM.btnStart.disabled = true;
    DOM.btnStart.textContent = '⏳ Initializing...';

    try {
        // 1. Start camera
        const cameraOk = await startCamera();
        if (!cameraOk) {
            DOM.btnStart.disabled = false;
            DOM.btnStart.textContent = '▶ Start Session';
            return;
        }

        // 2. Connect WebSocket
        await connectWebSocket();

        // 3. Start session on server
        sendMessage({ action: 'start_session' });

        // 4. Initialize UI
        STATE.sessionActive = true;
        STATE.sessionStartTime = Date.now();
        STATE.framesSent = 0;
        STATE.framesReceived = 0;
        STATE.fpsCounter = 0;
        STATE.fpsTimestamp = null;

        // Show UI elements
        DOM.cameraTimer.classList.remove('hidden');
        DOM.detectionInfo.classList.remove('hidden');
        DOM.phaseIndicator.classList.remove('hidden');
        DOM.progressSection.classList.remove('hidden');
        DOM.statsRow.classList.remove('hidden');
        DOM.scanLine.classList.add('active');
        DOM.progressFill.classList.add('active');

        updateStatus('recording', 'Recording');
        DOM.sessionBadge.className = 'card-badge badge-phase1';
        DOM.sessionBadge.textContent = 'PHASE 1 · STABILIZING';

        // Update button to Stop
        DOM.btnStart.textContent = '⏹ Stop Session';
        DOM.btnStart.className = 'btn-primary btn-stop';
        DOM.btnStart.disabled = false;

        // 5. Start frame capture (the Canvas trick)
        startCapture();

    } catch (err) {
        console.error('Failed to start session:', err);
        DOM.btnStart.disabled = false;
        DOM.btnStart.textContent = '▶ Start Session';
        updateStatus('error', 'Connection Failed');
    }
}

function endSession() {
    STATE.sessionActive = false;
    stopCapture();
    sendMessage({ action: 'end_session' });
    DOM.scanLine.classList.remove('active');
    updateStatus('active', 'Stopped');

    DOM.sessionBadge.className = 'card-badge badge-idle';
    DOM.sessionBadge.textContent = 'STOPPED';
    DOM.progressFill.classList.remove('active');

    DOM.btnStart.textContent = '🔄 New Session';
    DOM.btnStart.className = 'btn-primary btn-reset';
}


// =============================================================================
// OFFLINE FILE UPLOAD HANDLING
// =============================================================================
async function handleFileUpload(event) {
    const files = event.target.files;
    if (files.length === 0) return;

    // Stop current session if any
    if (STATE.sessionActive) {
        endSession();
        await new Promise(r => setTimeout(r, 300));
    } else {
        resetUI();
        stopCamera();
    }

    // Set UI to loading state
    DOM.hgWaiting.classList.remove('hidden');
    DOM.hgPreview.classList.add('hidden');
    DOM.finalResultCard.classList.remove('visible');
    DOM.statsRow.classList.remove('hidden');
    
    // Custom loading message
    DOM.hgWaiting.innerHTML = `
        <div class="hg-waiting">
            <div class="hg-waiting-dots">
                <span></span><span></span><span></span>
            </div>
            <span class="hg-label">Analyzing last 5 seconds of the uploaded file(s)...</span>
        </div>
    `;

    DOM.resultBadge.className = 'card-badge badge-phase2';
    DOM.resultBadge.textContent = 'ANALYZING';

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    try {
        const response = await fetch('/api/upload_offline', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        if (result.status === 'success') {
            // Restore original waiting HTML just in case
            DOM.hgWaiting.innerHTML = `
                <div class="hg-waiting">
                    <div class="hg-waiting-dots">
                        <span></span><span></span><span></span>
                    </div>
                    <span class="hg-label">Start the session to begin estimation</span>
                </div>
            `;
            showFinalResult(result.hg_final, result.frames_used);
        } else {
            alert('Upload Analysis Error: ' + result.message);
            updateStatus('error', 'Analysis failed');
            DOM.resultBadge.className = 'card-badge badge-error';
            DOM.resultBadge.textContent = 'ERROR';
        }
    } catch (err) {
        alert('Upload failed: ' + err);
        updateStatus('error', 'Upload failed');
    }
    
    // Clear the file input so the same files can be chosen again if needed
    event.target.value = '';
}


// =============================================================================
// INIT
// =============================================================================
console.log('[TABBA Web] Initialized. Ready to capture at 3 FPS via Canvas.');
