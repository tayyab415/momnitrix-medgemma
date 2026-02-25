import SwiftUI

struct ResultsView: View {
    @Environment(TriageViewModel.self) var vm
    @State private var bannerAppeared = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 12) {

                    // ── Risk level banner ──
                    RiskBanner(level: vm.riskLevel, appeared: bannerAppeared)

                    // ── Specialist outputs (voice transcript, wound, skin) ──
                    if let specialist = vm.finalResponse?.specialist_outputs {
                        if let transcript = specialist.transcript, !transcript.isEmpty {
                            ResultCard(title: "Voice Transcript", icon: "mic.fill", accentColor: .mint) {
                                Text(transcript)
                                    .font(.system(size: 10))
                                    .italic()
                            }
                        }

                        if let wound = specialist.wound_scores {
                            ResultCard(title: "Wound Analysis", icon: "bandage.fill", accentColor: .orange) {
                                WoundScoreRow(label: "Urgency", value: wound.urgency)
                                WoundScoreRow(label: "Infection Risk", value: wound.infection_risk)
                                WoundScoreRow(label: "Erythema", value: wound.erythema)
                            }
                        }

                        if let skinItems = specialist.skin_top3, !skinItems.isEmpty {
                            ResultCard(title: "Skin Analysis", icon: "hand.raised.fill", accentColor: .indigo) {
                                ForEach(skinItems, id: \.condition) { item in
                                    HStack {
                                        Text(item.condition)
                                            .font(.system(size: 10))
                                        Spacer()
                                        Text(String(format: "%.0f%%", item.score * 100))
                                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                                            .foregroundStyle(.indigo)
                                    }
                                }
                            }
                        }
                    }

                    // ── Action items ──
                    if !vm.actionItems.isEmpty {
                        ResultCard(title: "Actions", icon: "checklist", accentColor: .pink) {
                            ForEach(Array(vm.actionItems.enumerated()), id: \.offset) { idx, item in
                                HStack(alignment: .top, spacing: 6) {
                                    Text("\(idx + 1)")
                                        .font(.system(size: 9, weight: .bold, design: .rounded))
                                        .foregroundStyle(.white)
                                        .frame(width: 16, height: 16)
                                        .background(Circle().fill(.pink))
                                    Text(item)
                                        .font(.system(size: 10))
                                }
                            }
                        }
                    }

                    // ── Patient message ──
                    if !vm.patientMessage.isEmpty {
                        ResultCard(title: "Guidance", icon: "text.bubble.fill", accentColor: .cyan) {
                            Text(vm.patientMessage)
                                .font(.system(size: 10))
                        }
                    }

                    // ── Clinical reasons ──
                    if !vm.medgemmaReasons.isEmpty {
                        ResultCard(title: "Clinical Reasoning", icon: "brain.fill", accentColor: .purple) {
                            ForEach(vm.medgemmaReasons, id: \.self) { reason in
                                HStack(alignment: .top, spacing: 4) {
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 7))
                                        .foregroundStyle(.purple)
                                        .padding(.top, 2)
                                    Text(reason)
                                        .font(.system(size: 10))
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }

                    // ── Visit prep ──
                    if !vm.visitSummary.isEmpty {
                        ResultCard(title: "Visit Prep", icon: "doc.text.fill", accentColor: .blue) {
                            Text(vm.visitSummary)
                                .font(.system(size: 10))
                        }
                    }

                    // ── Latency info ──
                    if let latency = vm.totalLatencyMs {
                        HStack(spacing: 4) {
                            Image(systemName: "clock.fill")
                                .font(.system(size: 8))
                                .foregroundStyle(.tertiary)
                            Text("Total: \(Int(latency))ms")
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundStyle(.tertiary)
                        }
                    }

                    // ── New run button ──
                    Button {
                        vm.reset()
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "arrow.counterclockwise")
                                .font(.system(size: 11))
                            Text("New Triage")
                                .font(.system(size: 12, weight: .semibold))
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.pink)
                    .padding(.top, 4)
                }
                .padding(.horizontal, 6)
                .padding(.bottom, 16)
            }
            .navigationTitle("Results")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                withAnimation(.spring(response: 0.6, dampingFraction: 0.7)) {
                    bannerAppeared = true
                }
            }
        }
    }
}

// MARK: - Risk Banner

struct RiskBanner: View {
    let level: RiskLevel
    let appeared: Bool

    private var bgGradient: LinearGradient {
        switch level {
        case .green:
            return LinearGradient(colors: [.green, .green.opacity(0.7)], startPoint: .topLeading, endPoint: .bottomTrailing)
        case .yellow:
            return LinearGradient(colors: [.yellow, .orange.opacity(0.7)], startPoint: .topLeading, endPoint: .bottomTrailing)
        case .red:
            return LinearGradient(colors: [.red, .red.opacity(0.7)], startPoint: .topLeading, endPoint: .bottomTrailing)
        case .unknown:
            return LinearGradient(colors: [.gray, .gray.opacity(0.7)], startPoint: .topLeading, endPoint: .bottomTrailing)
        }
    }

    private var label: String {
        switch level {
        case .green:   return "LOW RISK"
        case .yellow:  return "MODERATE RISK"
        case .red:     return "HIGH RISK"
        case .unknown: return "UNKNOWN"
        }
    }

    private var subtitle: String {
        switch level {
        case .green:   return "Continue routine monitoring"
        case .yellow:  return "Schedule follow-up soon"
        case .red:     return "Seek immediate medical care"
        case .unknown: return "Unable to determine risk"
        }
    }

    var body: some View {
        VStack(spacing: 4) {
            Text(level.emoji)
                .font(.system(size: 28))
                .scaleEffect(appeared ? 1.0 : 0.3)

            Text(label)
                .font(.system(size: 15, weight: .black, design: .rounded))
                .foregroundStyle(level == .yellow ? .black : .white)

            Text(subtitle)
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(level == .yellow ? .black.opacity(0.7) : .white.opacity(0.8))
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
        .background(bgGradient)
        .cornerRadius(14)
        .shadow(color: level == .red ? .red.opacity(0.4) : .clear, radius: 8, y: 2)
    }
}

// MARK: - Result Card

struct ResultCard<Content: View>: View {
    let title: String
    var icon: String = ""
    var accentColor: Color = .pink
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 4) {
                if !icon.isEmpty {
                    Image(systemName: icon)
                        .font(.system(size: 9))
                        .foregroundStyle(accentColor)
                }
                Text(title)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(accentColor)
            }
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.white.opacity(0.06))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(accentColor.opacity(0.15), lineWidth: 1)
        )
    }
}

// MARK: - Wound Score Row

struct WoundScoreRow: View {
    let label: String
    let value: Double

    private var barColor: Color {
        if value >= 0.7 { return .red }
        if value >= 0.4 { return .orange }
        return .green
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label)
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.0f%%", value * 100))
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color.white.opacity(0.1))
                        .frame(height: 4)
                    Capsule()
                        .fill(barColor)
                        .frame(width: geo.size.width * CGFloat(min(value, 1.0)), height: 4)
                }
            }
            .frame(height: 4)
        }
    }
}
