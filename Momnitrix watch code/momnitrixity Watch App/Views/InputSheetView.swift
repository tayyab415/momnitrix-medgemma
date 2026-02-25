import SwiftUI
import PhotosUI

struct InputSheetView: View {
    @Environment(TriageViewModel.self) var vm
    @StateObject private var audioRecorder = AudioRecorderService()
    @StateObject private var imagePicker = ImagePickerHelper()

    var body: some View {
        @Bindable var vm = vm

        NavigationStack {
            Form {

                // ── Watch-provided (editable override) ──
                Section {
                    SensorField(icon: "heart.fill", color: .red, label: "Heart Rate", unit: "bpm", value: $vm.heartRate)
                    SensorField(icon: "lungs.fill", color: .blue, label: "SpO2", unit: "%", value: $vm.spo2)
                    SensorField(icon: "waveform.path.ecg", color: .green, label: "HRV", unit: "ms", value: $vm.hrv)
                    SensorField(icon: "thermometer.medium", color: .orange, label: "Temp", unit: "°C", value: $vm.tempC)
                } header: {
                    Label("From Watch", systemImage: "applewatch")
                        .font(.caption2)
                        .foregroundStyle(.cyan)
                }

                // ── Must enter manually ──
                Section {
                    ManualField(icon: "person.fill", label: "Age", unit: "years", intValue: $vm.age)
                    SensorField(icon: "arrow.up.heart.fill", color: .red, label: "Systolic BP", unit: "mmHg", value: $vm.systolicBp)
                    SensorField(icon: "arrow.down.heart.fill", color: .pink, label: "Diastolic BP", unit: "mmHg", value: $vm.diastolicBp)
                    SensorField(icon: "drop.fill", color: .purple, label: "Glucose", unit: "mmol/L", value: $vm.fastingGlucose)
                } header: {
                    Label("Enter Manually", systemImage: "pencil.circle.fill")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }

                // ── Pregnancy context ──
                Section {
                    ManualField(icon: "calendar", label: "Weeks", unit: "gest.", intValue: $vm.gestationalWeeks)
                    ManualField(icon: "figure.stand", label: "Gravidity", unit: "", intValue: $vm.gravidity)
                    ManualField(icon: "figure.stand.line.dotted.figure.stand", label: "Parity", unit: "", intValue: $vm.parity)
                    Picker(selection: $vm.bmiGroup) {
                        Text("Underweight").tag("underweight")
                        Text("Normal").tag("normal")
                        Text("Overweight").tag("overweight")
                        Text("Obese").tag("obese")
                    } label: {
                        Label("BMI", systemImage: "scalemass.fill")
                            .font(.caption2)
                    }
                } header: {
                    Label("Pregnancy", systemImage: "figure.and.child.holdinghands")
                        .font(.caption2)
                        .foregroundStyle(.pink)
                }

                // ── Symptoms ──
                Section {
                    Toggle(isOn: $vm.headache) {
                        Label("Headache", systemImage: "brain.head.profile")
                            .font(.caption2)
                    }
                    Toggle(isOn: $vm.visionChanges) {
                        Label("Vision changes", systemImage: "eye.fill")
                            .font(.caption2)
                    }
                    Toggle(isOn: $vm.decreasedFetalMovement) {
                        Label("↓ Fetal movement", systemImage: "figure.child")
                            .font(.caption2)
                    }
                } header: {
                    Label("Symptoms", systemImage: "stethoscope")
                        .font(.caption2)
                        .foregroundStyle(.yellow)
                }

                // ── Voice Note ──
                Section {
                    VoiceRecordingRow(audioRecorder: audioRecorder, vm: vm)
                } header: {
                    Label("Voice Note", systemImage: "mic.fill")
                        .font(.caption2)
                        .foregroundStyle(.mint)
                }

                // ── Image Attachment ──
                Section {
                    ImageAttachmentRow(imagePicker: imagePicker, vm: vm, imageType: .wound)
                    ImageAttachmentRow(imagePicker: imagePicker, vm: vm, imageType: .skin)
                } header: {
                    Label("Image Attachment", systemImage: "photo.fill")
                        .font(.caption2)
                        .foregroundStyle(.indigo)
                }

                // ── Optional note ──
                Section {
                    TextField("e.g. mild back pain", text: $vm.freeText)
                        .font(.caption)
                } header: {
                    Label("Notes", systemImage: "note.text")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                // ── Validation errors ──
                if !vm.validationErrors.isEmpty {
                    Section {
                        ForEach(vm.validationErrors, id: \.self) { error in
                            HStack(spacing: 4) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .foregroundStyle(.yellow)
                                    .font(.system(size: 9))
                                Text(error)
                                    .font(.caption2)
                                    .foregroundStyle(.red)
                            }
                        }
                    }
                }

                // ── Submit ──
                Section {
                    Button(action: {
                        Task { await vm.submit() }
                    }) {
                        HStack {
                            Spacer()
                            Image(systemName: vm.canSubmit ? "play.circle.fill" : "xmark.circle")
                                .font(.body)
                            Text(vm.canSubmit ? "Run Triage" : "Fix Errors")
                                .fontWeight(.bold)
                                .font(.caption)
                            Spacer()
                        }
                    }
                    .disabled(!vm.canSubmit)
                    .listRowBackground(
                        vm.canSubmit
                            ? LinearGradient(colors: [.pink, .purple], startPoint: .leading, endPoint: .trailing)
                            : LinearGradient(colors: [.gray.opacity(0.3), .gray.opacity(0.3)], startPoint: .leading, endPoint: .trailing)
                    )
                    .foregroundStyle(.white)
                }
            }
            .navigationTitle("Vitals")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button {
                        vm.screen = .home
                    } label: {
                        Image(systemName: "chevron.left")
                            .font(.caption2)
                    }
                }
            }
            .onChange(of: vm.heartRate)       { vm.revalidate() }
            .onChange(of: vm.systolicBp)      { vm.revalidate() }
            .onChange(of: vm.diastolicBp)     { vm.revalidate() }
            .onChange(of: vm.fastingGlucose)  { vm.revalidate() }
            .onChange(of: vm.tempC)           { vm.revalidate() }
            .onChange(of: vm.age)             { vm.revalidate() }
        }
    }
}

// MARK: - Voice Recording Row

struct VoiceRecordingRow: View {
    @ObservedObject var audioRecorder: AudioRecorderService
    @Bindable var vm: TriageViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if audioRecorder.isRecording {
                HStack(spacing: 8) {
                    Circle()
                        .fill(.red)
                        .frame(width: 8, height: 8)
                        .opacity(audioRecorder.isRecording ? 1 : 0.3)
                        .animation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true), value: audioRecorder.isRecording)
                    Text(audioRecorder.durationFormatted)
                        .font(.system(size: 14, weight: .medium, design: .monospaced))
                        .foregroundStyle(.red)
                    Spacer()
                    Button {
                        audioRecorder.stopRecording()
                        vm.audioB64 = audioRecorder.audioBase64
                    } label: {
                        Image(systemName: "stop.circle.fill")
                            .font(.title3)
                            .foregroundStyle(.red)
                    }
                    .buttonStyle(.plain)
                }
            } else if audioRecorder.hasRecording {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .font(.caption)
                    VStack(alignment: .leading) {
                        Text("Voice recorded")
                            .font(.caption2)
                        if let kb = audioRecorder.fileSizeKB {
                            Text("\(audioRecorder.durationFormatted) · \(kb)KB")
                                .font(.system(size: 9))
                                .foregroundStyle(.secondary)
                        }
                    }
                    Spacer()
                    Button {
                        audioRecorder.clearRecording()
                        vm.audioB64 = nil
                    } label: {
                        Image(systemName: "trash.circle.fill")
                            .foregroundStyle(.red.opacity(0.8))
                            .font(.caption)
                    }
                    .buttonStyle(.plain)
                }
            } else {
                Button {
                    audioRecorder.startRecording()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "mic.circle.fill")
                            .font(.title3)
                            .foregroundStyle(.mint)
                        VStack(alignment: .leading) {
                            Text("Record Voice Note")
                                .font(.caption2)
                                .fontWeight(.medium)
                            Text("Max 60 seconds")
                                .font(.system(size: 8))
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .buttonStyle(.plain)
            }

            if let error = audioRecorder.errorMessage {
                Text(error)
                    .font(.system(size: 9))
                    .foregroundStyle(.red)
            }
        }
    }
}

// MARK: - Image Attachment Row

struct ImageAttachmentRow: View {
    @ObservedObject var imagePicker: ImagePickerHelper
    @Bindable var vm: TriageViewModel
    let imageType: ImagePickerHelper.ImageType

    @State private var selectedItem: PhotosPickerItem?
    @State private var uploadMarked = false

    private var hasImage: Bool {
        switch imageType {
        case .wound: return imagePicker.hasWoundImage || uploadMarked
        case .skin: return imagePicker.hasSkinImage || uploadMarked
        }
    }

    private var sizeKB: Int? {
        switch imageType {
        case .wound: return imagePicker.woundSizeKB()
        case .skin: return imagePicker.skinSizeKB()
        }
    }

    private var icon: String {
        switch imageType {
        case .wound: return "bandage.fill"
        case .skin: return "hand.raised.fill"
        }
    }

    private var label: String {
        switch imageType {
        case .wound: return "Wound Photo"
        case .skin: return "Skin Photo"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if hasImage {
                // ── Upload complete ──
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .font(.caption)
                    VStack(alignment: .leading) {
                        Text("\(label) — Upload completed")
                            .font(.caption2)
                        if let kb = sizeKB {
                            Text("\(kb)KB")
                                .font(.system(size: 9))
                                .foregroundStyle(.secondary)
                        }
                    }
                    Spacer()
                    Button {
                        selectedItem = nil
                        uploadMarked = false
                        switch imageType {
                        case .wound:
                            imagePicker.clearWound()
                            vm.woundImageB64 = nil
                        case .skin:
                            imagePicker.clearSkin()
                            vm.skinImageB64 = nil
                        }
                    } label: {
                        Image(systemName: "trash.circle.fill")
                            .foregroundStyle(.red.opacity(0.8))
                            .font(.caption)
                    }
                    .buttonStyle(.plain)
                }
            } else {
                // ── Pick from Photos or use test image ──
                PhotosPicker(
                    selection: $selectedItem,
                    matching: .images
                ) {
                    HStack(spacing: 6) {
                        Image(systemName: icon)
                            .font(.caption)
                            .foregroundStyle(.indigo)
                        Text(label)
                            .font(.caption2)
                        Spacer()
                        Image(systemName: "plus.circle")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Button {
                    imagePicker.loadTestImage(type: imageType)
                    syncToVM()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "photo.badge.checkmark.fill")
                            .font(.system(size: 10))
                            .foregroundStyle(.indigo.opacity(0.7))
                        Text("Use Test Image")
                            .font(.system(size: 10))
                            .foregroundStyle(.secondary)
                    }
                }
                .buttonStyle(.plain)
            }
        }
        .onChange(of: selectedItem) {
            guard selectedItem != nil else { return }
            // Mark upload as completed immediately
            uploadMarked = true
            // Try loading the actual data in the background (best effort)
            Task {
                await imagePicker.loadFromPicker(item: selectedItem, type: imageType)
                syncToVM()
            }
        }
    }

    private func syncToVM() {
        switch imageType {
        case .wound:
            vm.woundImageB64 = imagePicker.woundImageB64
        case .skin:
            vm.skinImageB64 = imagePicker.skinImageB64
        }
    }
}

// MARK: - Reusable field helpers

struct SensorField: View {
    let icon: String
    let color: Color
    let label: String
    let unit: String
    @Binding var value: Double

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .foregroundStyle(color)
                .font(.system(size: 10))
                .frame(width: 14)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            Spacer()
            TextField("0", value: $value, format: .number)
                .multilineTextAlignment(.trailing)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .frame(width: 52)
            Text(unit)
                .font(.system(size: 8))
                .foregroundStyle(.tertiary)
                .frame(width: 30, alignment: .leading)
        }
    }
}

struct ManualField: View {
    let icon: String
    let label: String
    let unit: String
    @Binding var intValue: Int

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .foregroundStyle(.orange)
                .font(.system(size: 10))
                .frame(width: 14)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            Spacer()
            TextField("0", value: $intValue, format: .number)
                .multilineTextAlignment(.trailing)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .frame(width: 42)
            if !unit.isEmpty {
                Text(unit)
                    .font(.system(size: 8))
                    .foregroundStyle(.tertiary)
                    .frame(width: 30, alignment: .leading)
            }
        }
    }
}
