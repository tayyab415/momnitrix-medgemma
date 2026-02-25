import SwiftUI

struct HomeView: View {
    @Environment(TriageViewModel.self) var vm
    @StateObject private var healthReader = HealthReader()
    @State private var pulseAnimation = false

    var body: some View {
        @Bindable var vm = vm

        ScrollView {
            VStack(spacing: 14) {

                // ── App Header with gradient ──
                VStack(spacing: 4) {
                    Image(systemName: "heart.text.clipboard.fill")
                        .font(.system(size: 28))
                        .foregroundStyle(
                            LinearGradient(colors: [.pink, .purple], startPoint: .topLeading, endPoint: .bottomTrailing)
                        )
                        .scaleEffect(pulseAnimation ? 1.05 : 1.0)
                        .animation(.easeInOut(duration: 1.5).repeatForever(autoreverses: true), value: pulseAnimation)

                    Text("Momnitrix")
                        .font(.system(size: 18, weight: .bold, design: .rounded))
                        .foregroundStyle(
                            LinearGradient(colors: [.pink, .purple], startPoint: .leading, endPoint: .trailing)
                        )

                    Text("Maternal Health Triage")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                // ── Backend status ──
                HStack(spacing: 5) {
                    Circle()
                        .fill(vm.backendHealthy ? Color.green : Color.red)
                        .frame(width: 6, height: 6)
                        .shadow(color: vm.backendHealthy ? .green.opacity(0.6) : .red.opacity(0.6), radius: 3)
                    Text(vm.backendHealthy ? "Backend Online" : "Backend Offline")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(vm.backendHealthy ? .green : .red)
                }
                .padding(.vertical, 3)
                .padding(.horizontal, 10)
                .background(.ultraThinMaterial)
                .clipShape(Capsule())

                // ── Sensor Cards Grid ──
                VStack(spacing: 8) {
                    HStack(spacing: 8) {
                        SensorCard(
                            icon: "heart.fill",
                            label: "Heart Rate",
                            value: healthReader.heartRateDisplay,
                            color: .red,
                            isAvailable: healthReader.heartRate != nil
                        )
                        SensorCard(
                            icon: "lungs.fill",
                            label: "SpO2",
                            value: healthReader.spo2Display,
                            color: .cyan,
                            isAvailable: healthReader.spo2 != nil,
                            isHighlighted: true  // SpO2 prominent
                        )
                    }
                    HStack(spacing: 8) {
                        SensorCard(
                            icon: "waveform.path.ecg",
                            label: "HRV",
                            value: healthReader.hrvDisplay,
                            color: .green,
                            isAvailable: healthReader.hrv != nil
                        )
                        SensorCard(
                            icon: "thermometer.medium",
                            label: "Temp",
                            value: healthReader.tempDisplay,
                            color: .orange,
                            isAvailable: healthReader.wristTempC != nil
                        )
                    }
                }

                if let error = healthReader.permissionError {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 9))
                            .foregroundStyle(.yellow)
                        Text(error)
                            .font(.system(size: 8))
                            .foregroundStyle(.orange)
                    }
                    .padding(6)
                    .background(Color.orange.opacity(0.1))
                    .cornerRadius(6)
                }

                if healthReader.heartRate == nil && healthReader.spo2 == nil {
                    Text("No recent sensor data — defaults will be used")
                        .font(.system(size: 8))
                        .foregroundStyle(.orange)
                        .multilineTextAlignment(.center)
                }

                // ── Refresh button ──
                Button {
                    Task { await healthReader.readAll() }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.clockwise")
                            .font(.system(size: 10))
                        Text("Refresh Sensors")
                            .font(.system(size: 11, weight: .medium))
                    }
                }
                .buttonStyle(.bordered)
                .tint(.cyan)

                // ── Main action ──
                Button {
                    vm.populateFromHealthKit(healthReader)
                    vm.screen = .inputSheet
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "stethoscope")
                            .font(.system(size: 13))
                        Text("Start Triage")
                            .font(.system(size: 13, weight: .bold))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                }
                .buttonStyle(.borderedProminent)
                .tint(.pink)
            }
            .padding(.horizontal, 8)
            .padding(.bottom, 12)
        }
        .task {
            pulseAnimation = true
            await healthReader.requestPermission()
            await vm.checkBackendHealth()
        }
    }
}

// MARK: - Sensor Card

struct SensorCard: View {
    let icon: String
    let label: String
    let value: String
    let color: Color
    let isAvailable: Bool
    var isHighlighted: Bool = false

    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: isHighlighted ? 16 : 13))
                .foregroundStyle(color)

            Text(value)
                .font(.system(size: isHighlighted ? 14 : 12, weight: .bold, design: .rounded))
                .foregroundStyle(isAvailable ? .primary : .secondary)

            Text(label)
                .font(.system(size: 8, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(
                    isHighlighted
                    ? LinearGradient(colors: [color.opacity(0.2), color.opacity(0.05)], startPoint: .top, endPoint: .bottom)
                    : LinearGradient(colors: [Color.white.opacity(0.08), Color.white.opacity(0.03)], startPoint: .top, endPoint: .bottom)
                )
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(isHighlighted ? color.opacity(0.4) : Color.white.opacity(0.1), lineWidth: 1)
        )
    }
}

// MARK: - SensorRow (kept for backward compatibility)

struct SensorRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).fontWeight(.medium)
        }
    }
}
