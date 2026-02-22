// Pulse recording indicator — floating pill overlay.
//
// Communicates via newline-delimited commands on stdin:
//   "show\n"         — fade in the pill
//   "hide\n"         — fade out the pill
//   "level 0.42\n"   — update audio level (0.0–1.0) for waveform
//
// Runs as an accessory process (no Dock icon, no activation).

import AppKit

// MARK: - Indicator Window

final class IndicatorWindow {
    private let panel: NSPanel
    private let dot: NSView
    private let waveformView: WaveformView
    private var pulseTimer: Timer?

    init() {
        let pillWidth: CGFloat = 120
        let pillHeight: CGFloat = 28

        // Container pill view
        let pill = NSView(frame: NSRect(x: 0, y: 0, width: pillWidth, height: pillHeight))
        pill.wantsLayer = true
        pill.layer?.backgroundColor = NSColor.black.withAlphaComponent(0.7).cgColor
        pill.layer?.cornerRadius = pillHeight / 2

        // Subtle shadow on the pill layer
        pill.shadow = NSShadow()
        pill.layer?.shadowColor = NSColor.black.withAlphaComponent(0.3).cgColor
        pill.layer?.shadowOffset = CGSize(width: 0, height: -1)
        pill.layer?.shadowRadius = 8
        pill.layer?.shadowOpacity = 1

        // Red recording dot
        let dotSize: CGFloat = 7
        dot = NSView(frame: NSRect(x: 11, y: (pillHeight - dotSize) / 2, width: dotSize, height: dotSize))
        dot.wantsLayer = true
        dot.layer?.backgroundColor = NSColor(red: 1.0, green: 0.28, blue: 0.28, alpha: 1.0).cgColor
        dot.layer?.cornerRadius = dotSize / 2
        pill.addSubview(dot)

        // Waveform bars — compact, centered in remaining space
        let waveX: CGFloat = 24
        let waveWidth: CGFloat = pillWidth - waveX - 10
        waveformView = WaveformView(frame: NSRect(x: waveX, y: 5, width: waveWidth, height: 18))
        pill.addSubview(waveformView)

        // Panel setup
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: pillWidth, height: pillHeight),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.isFloatingPanel = true
        panel.level = .statusBar
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.hidesOnDeactivate = false
        panel.ignoresMouseEvents = true
        panel.contentView = pill
        panel.alphaValue = 0

        self.panel = panel
        positionWindow()
    }

    func show() {
        positionWindow()
        panel.orderFrontRegardless()

        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.25
            ctx.timingFunction = CAMediaTimingFunction(name: .easeOut)
            panel.animator().alphaValue = 1
        }

        // Start pulsing dot
        pulseTimer = Timer.scheduledTimer(withTimeInterval: 0.8, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            let current = self.dot.layer?.opacity ?? 1.0
            let target: Float = current < 0.9 ? 1.0 : 0.4
            NSAnimationContext.runAnimationGroup { ctx in
                ctx.duration = 0.8
                self.dot.animator().alphaValue = CGFloat(target)
            }
        }
    }

    func hide() {
        pulseTimer?.invalidate()
        pulseTimer = nil

        NSAnimationContext.runAnimationGroup({ ctx in
            ctx.duration = 0.25
            ctx.timingFunction = CAMediaTimingFunction(name: .easeIn)
            panel.animator().alphaValue = 0
        }, completionHandler: {
            self.panel.orderOut(nil)
            self.waveformView.setLevel(0)
        })
    }

    func setLevel(_ level: Float) {
        waveformView.setLevel(level)
    }

    private func positionWindow() {
        guard let screen = NSScreen.main else { return }
        let visibleFrame = screen.visibleFrame
        let size = panel.frame.size
        let padding: CGFloat = 16
        let origin = CGPoint(
            x: visibleFrame.midX - size.width / 2,
            y: visibleFrame.minY + padding
        )
        panel.setFrameOrigin(origin)
    }
}

// MARK: - Waveform View (vertical bars)

final class WaveformView: NSView {
    private let barCount = 24
    private var barLayers: [CALayer] = []
    private var currentLevel: Float = 0
    private var barHeights: [CGFloat]
    private var animTimer: Timer?

    override init(frame: NSRect) {
        barHeights = Array(repeating: 0, count: barCount)
        super.init(frame: frame)
        wantsLayer = true

        let barWidth: CGFloat = 2
        let gap: CGFloat = (frame.width - CGFloat(barCount) * barWidth) / CGFloat(barCount - 1)

        for i in 0..<barCount {
            let bar = CALayer()
            let x = CGFloat(i) * (barWidth + gap)
            // Start as a tiny centered line
            bar.frame = CGRect(x: x, y: frame.height / 2 - 0.5, width: barWidth, height: 1)
            bar.backgroundColor = NSColor.white.withAlphaComponent(0.9).cgColor
            bar.cornerRadius = barWidth / 2
            layer?.addSublayer(bar)
            barLayers.append(bar)
        }

        // ~60fps for smooth bar animation
        animTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            self?.tick()
        }
        RunLoop.main.add(animTimer!, forMode: .common)
    }

    required init?(coder: NSCoder) { fatalError() }

    deinit {
        animTimer?.invalidate()
    }

    func setLevel(_ level: Float) {
        // Boost raw RMS into visible range
        currentLevel = min(level * 8.0, 1.0)
    }

    private func tick() {
        let level = CGFloat(currentLevel)
        let maxH = bounds.height
        let barWidth: CGFloat = 2

        CATransaction.begin()
        CATransaction.setDisableActions(true)

        for (i, bar) in barLayers.enumerated() {
            // Each bar has its own target with randomness
            let target: CGFloat
            if level < 0.01 {
                target = 1  // idle: tiny line
            } else {
                // Center bars taller, edges shorter (bell curve envelope)
                let center = CGFloat(barCount - 1) / 2.0
                let dist = abs(CGFloat(i) - center) / center
                let envelope = 1.0 - dist * 0.6
                target = max(1, level * envelope * CGFloat.random(in: 0.3...1.0) * maxH)
            }

            // Smooth toward target — fast rise, slower fall for punchy feel
            let current = barHeights[i]
            if target > current {
                barHeights[i] += (target - current) * 0.4  // fast attack
            } else {
                barHeights[i] += (target - current) * 0.15  // slow decay
            }

            let h = barHeights[i]
            let x = bar.frame.origin.x
            bar.frame = CGRect(x: x, y: (maxH - h) / 2, width: barWidth, height: h)
        }

        CATransaction.commit()
    }
}

// MARK: - Main

let app = NSApplication.shared
app.setActivationPolicy(.accessory)

let indicator = IndicatorWindow()

DispatchQueue.global(qos: .userInteractive).async {
    while let line = readLine() {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        switch trimmed {
        case "show":
            DispatchQueue.main.async { indicator.show() }
        case "hide":
            DispatchQueue.main.async { indicator.hide() }
        case "quit":
            DispatchQueue.main.async { app.terminate(nil) }
        case let cmd where cmd.hasPrefix("level "):
            let parts = cmd.split(separator: " ", maxSplits: 1)
            if parts.count == 2, let level = Float(parts[1]) {
                DispatchQueue.main.async { indicator.setLevel(level) }
            }
        default:
            break
        }
    }
    DispatchQueue.main.async { app.terminate(nil) }
}

app.run()
