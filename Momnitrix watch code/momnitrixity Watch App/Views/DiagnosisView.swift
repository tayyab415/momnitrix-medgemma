import SwiftUI

struct DiagnosisView: View {
    @Environment(TriageViewModel.self) var vm
    @State private var dotOpacity: Double = 0.3

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 10) {

                    // ── Animated header ──
                    if vm.isStreaming {
                        VStack(spacing: 8) {
                            ZStack {
                                Circle()
                                    .stroke(Color.pink.opacity(0.2), lineWidth: 3)
                                    .frame(width: 44, height: 44)
                                Circle()
                                    .trim(from: 0, to: 0.7)
                                    .stroke(
                                        LinearGradient(colors: [.pink, .purple], startPoint: .leading, endPoint: .trailing),
                                        style: StrokeStyle(lineWidth: 3, lineCap: .round)
                                    )
                                    .frame(width: 44, height: 44)
                                    .rotationEffect(.degrees(dotOpacity > 0.5 ? 360 : 0))
                                    .animation(.linear(duration: 1.5).repeatForever(autoreverses: false), value: dotOpacity)
                                Image(systemName: "brain.head.profile")
                                    .font(.system(size: 18))
                                    .foregroundStyle(.pink)
                            }

                            Text("Analyzing Your Vitals")
                                .font(.system(size: 12, weight: .semibold, design: .rounded))
                                .foregroundStyle(.pink)

                            Text("Streaming from MedGemma…")
                                .font(.system(size: 9))
                                .foregroundStyle(.secondary)
                        }
                        .padding(.top, 8)
                        .padding(.bottom, 4)
                    }

                    // ── Error banner ──
                    if let error = vm.errorMessage {
                        HStack(spacing: 4) {
                            Image(systemName: "exclamationmark.octagon.fill")
                                .foregroundStyle(.red)
                                .font(.system(size: 12))
                            Text(error)
                                .font(.system(size: 10))
                                .foregroundStyle(.red)
                        }
                        .padding(8)
                        .frame(maxWidth: .infinity)
                        .background(Color.red.opacity(0.1))
                        .cornerRadius(8)
                        .padding(.horizontal, 4)
                    }

                    // ── Event timeline ──
                    ForEach(Array(vm.events.enumerated()), id: \.element.id) { index, event in
                        TimelineRow(event: event, index: index, isLatest: index == vm.events.count - 1 && vm.isStreaming)
                    }

                    if vm.events.isEmpty && vm.isStreaming {
                        VStack(spacing: 6) {
                            ProgressView()
                                .scaleEffect(0.6)
                            Text("Waiting for response…")
                                .font(.system(size: 9))
                                .foregroundStyle(.tertiary)
                        }
                        .padding(.top, 20)
                    }
                }
                .padding(.horizontal, 6)
                .padding(.bottom, 16)
            }
            .navigationTitle("Diagnosing")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button {
                        vm.reset()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                            .font(.system(size: 14))
                    }
                }
            }
            .onAppear { dotOpacity = 1.0 }
        }
    }
}

// MARK: - Timeline Row

struct TimelineRow: View {
    let event: SSEEvent
    let index: Int
    let isLatest: Bool

    @State private var appeared = false

    private var icon: String {
        switch event.name {
        case "request.accepted":     return "checkmark.circle.fill"
        case "request.rejected":     return "xmark.octagon.fill"
        case "router.decision":      return "arrow.triangle.branch"
        case "router.prompt_plan":   return "doc.text.fill"
        case "artifact.uploaded":    return "arrow.up.circle.fill"
        case "model.started":        return "play.circle.fill"
        case "model.completed":      return "checkmark.seal.fill"
        case "model.failed":         return "xmark.seal.fill"
        case "medgemma.started":     return "brain"
        case "medgemma.completed":   return "brain.fill"
        case "medgemma.delta":       return "text.bubble.fill"
        case "gemini.started":       return "sparkles"
        case "gemini.completed":     return "sparkle"
        case "gemini.skipped":       return "forward.fill"
        case "diagnostics.inference_breakdown": return "chart.bar.fill"
        case "triage.final":         return "flag.checkered"
        case "triage.error":         return "exclamationmark.triangle.fill"
        default:                     return "circle.fill"
        }
    }

    private var iconColor: Color {
        switch event.name {
        case "request.accepted":                    return .green
        case "request.rejected", "model.failed",
             "triage.error":                        return .red
        case "medgemma.started", "medgemma.completed",
             "medgemma.delta":                      return .purple
        case "gemini.started", "gemini.completed":  return .blue
        case "triage.final":                        return .pink
        case "diagnostics.inference_breakdown":     return .orange
        default:                                    return .cyan
        }
    }

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            // Timeline dot and line
            VStack(spacing: 0) {
                Image(systemName: icon)
                    .font(.system(size: 11))
                    .foregroundStyle(iconColor)
                    .frame(width: 18, height: 18)

                if !isLatest || !isLatest {
                    Rectangle()
                        .fill(Color.white.opacity(0.1))
                        .frame(width: 1)
                        .frame(maxHeight: .infinity)
                }
            }

            // Event info
            VStack(alignment: .leading, spacing: 2) {
                Text(event.displayLabel)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(isLatest ? .primary : .secondary)

                if let detail = event.detail {
                    Text(detail)
                        .font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundStyle(iconColor.opacity(0.8))
                }
            }

            Spacer()

            // Pulse dot for latest
            if isLatest {
                Circle()
                    .fill(iconColor)
                    .frame(width: 5, height: 5)
                    .opacity(appeared ? 1 : 0.3)
                    .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: appeared)
            }
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 6)
        .background(
            isLatest
            ? RoundedRectangle(cornerRadius: 8).fill(iconColor.opacity(0.08))
            : RoundedRectangle(cornerRadius: 8).fill(Color.clear)
        )
        .onAppear { appeared = true }
    }
}
