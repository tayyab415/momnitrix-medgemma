import Foundation
import AVFoundation
import Combine

/// Records audio on Apple Watch, converts to base64 for the MedASR backend.
/// Uses AVAudioRecorder to capture m4a (AAC) at 16kHz mono — small payload, good quality.
@MainActor
final class AudioRecorderService: ObservableObject {

    @Published var isRecording = false
    @Published var recordingDuration: TimeInterval = 0
    @Published var audioBase64: String?
    @Published var errorMessage: String?

    static let maxDuration: TimeInterval = 60  // 60s max

    private var recorder: AVAudioRecorder?
    private var timer: Timer?
    private var fileURL: URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("momnitrix_voice.m4a")
    }

    // MARK: - Start / Stop

    func startRecording() {
        errorMessage = nil
        audioBase64 = nil
        recordingDuration = 0

        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.record, mode: .default)
            try session.setActive(true)
        } catch {
            errorMessage = "Audio session error: \(error.localizedDescription)"
            return
        }

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 16000,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
        ]

        do {
            // Remove any old recording
            try? FileManager.default.removeItem(at: fileURL)
            recorder = try AVAudioRecorder(url: fileURL, settings: settings)
            recorder?.record(forDuration: Self.maxDuration)
            isRecording = true
            startTimer()
        } catch {
            errorMessage = "Recorder error: \(error.localizedDescription)"
        }
    }

    func stopRecording() {
        recorder?.stop()
        isRecording = false
        stopTimer()

        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            errorMessage = "No recording file found"
            return
        }

        do {
            let data = try Data(contentsOf: fileURL)
            audioBase64 = data.base64EncodedString()
        } catch {
            errorMessage = "Could not read recording: \(error.localizedDescription)"
        }
    }

    func clearRecording() {
        audioBase64 = nil
        recordingDuration = 0
        errorMessage = nil
        try? FileManager.default.removeItem(at: fileURL)
    }

    // MARK: - Timer

    private func startTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, self.isRecording else { return }
                self.recordingDuration = self.recorder?.currentTime ?? 0
                if self.recordingDuration >= Self.maxDuration {
                    self.stopRecording()
                }
            }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    // MARK: - Helpers

    var durationFormatted: String {
        let secs = Int(recordingDuration)
        return String(format: "%d:%02d", secs / 60, secs % 60)
    }

    var hasRecording: Bool {
        audioBase64 != nil
    }

    var fileSizeKB: Int? {
        guard let b64 = audioBase64 else { return nil }
        return (b64.count * 3 / 4) / 1024
    }
}
