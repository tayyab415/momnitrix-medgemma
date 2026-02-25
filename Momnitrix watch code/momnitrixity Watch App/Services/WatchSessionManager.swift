import Foundation
import WatchConnectivity
import Combine

/// Manages WatchConnectivity session to receive images from the paired iPhone.
/// The iPhone companion app (or a share extension) sends images via WCSession.
/// This is the only reliable way to get photo data onto a watchOS app —
/// PhotosPicker on watchOS cannot deliver image data to third-party apps.
final class WatchSessionManager: NSObject, ObservableObject, @unchecked Sendable {

    static let shared = WatchSessionManager()

    @Published var receivedWoundImage: Data?
    @Published var receivedSkinImage: Data?
    @Published var isReachable: Bool = false
    @Published var lastMessage: String?

    private var session: WCSession?

    private override init() {
        super.init()
    }

    func activate() {
        guard WCSession.isSupported() else {
            lastMessage = "WCSession not supported"
            return
        }
        session = WCSession.default
        session?.delegate = self
        session?.activate()
    }

    /// Request the iPhone to open its photo picker and send an image
    func requestImageFromPhone(type: String) {
        guard let session, session.isReachable else {
            lastMessage = "iPhone not reachable"
            return
        }
        session.sendMessage(
            ["action": "requestImage", "type": type],
            replyHandler: { reply in
                DispatchQueue.main.async {
                    self.lastMessage = reply["status"] as? String ?? "Sent"
                }
            },
            errorHandler: { error in
                DispatchQueue.main.async {
                    self.lastMessage = "Error: \(error.localizedDescription)"
                }
            }
        )
        lastMessage = "Requesting \(type) image from iPhone…"
    }

    func clearWound() {
        receivedWoundImage = nil
    }

    func clearSkin() {
        receivedSkinImage = nil
    }
}

// MARK: - WCSessionDelegate

extension WatchSessionManager: WCSessionDelegate {

    nonisolated func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        DispatchQueue.main.async {
            self.isReachable = session.isReachable
            if let error {
                self.lastMessage = "WCSession error: \(error.localizedDescription)"
            } else {
                self.lastMessage = activationState == .activated ? "Connected to iPhone" : nil
            }
        }
    }

    nonisolated func sessionReachabilityDidChange(_ session: WCSession) {
        DispatchQueue.main.async {
            self.isReachable = session.isReachable
        }
    }

    /// Receive image data sent from the iPhone via userInfo or message
    nonisolated func session(_ session: WCSession, didReceiveMessageData messageData: Data, replyHandler: @escaping (Data) -> Void) {
        // iPhone sends raw image data — store it
        DispatchQueue.main.async {
            // Default to wound if no metadata
            self.receivedWoundImage = messageData
            self.lastMessage = "Received image (\(messageData.count / 1024)KB)"
        }
        replyHandler(Data("ok".utf8))
    }

    /// Receive image data with metadata (type: wound/skin)
    nonisolated func session(_ session: WCSession, didReceiveMessage message: [String: Any], replyHandler: @escaping ([String: Any]) -> Void) {
        DispatchQueue.main.async {
            if let imageB64 = message["image_b64"] as? String,
               let data = Data(base64Encoded: imageB64) {
                let type = message["type"] as? String ?? "wound"
                if type == "skin" {
                    self.receivedSkinImage = data
                } else {
                    self.receivedWoundImage = data
                }
                self.lastMessage = "Received \(type) image (\(data.count / 1024)KB)"
            }
            replyHandler(["status": "received"])
        }
    }

    /// Receive files transferred from iPhone
    nonisolated func session(_ session: WCSession, didReceive file: WCSessionFile) {
        let metadata = file.metadata
        let type = metadata?["type"] as? String ?? "wound"

        guard let data = try? Data(contentsOf: file.fileURL) else { return }

        DispatchQueue.main.async {
            if type == "skin" {
                self.receivedSkinImage = data
            } else {
                self.receivedWoundImage = data
            }
            self.lastMessage = "Received \(type) image file (\(data.count / 1024)KB)"
        }
    }
}
