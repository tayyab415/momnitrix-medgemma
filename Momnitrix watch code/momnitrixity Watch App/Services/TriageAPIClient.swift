import Foundation

// Calls POST /v1/triage/stream and streams SSE events back.

@MainActor
final class TriageAPIClient {

    static let baseURL = "https://tayyabkhn343--momnitrix-api-v2-web.modal.run"

    private let session: URLSession = .shared
    private let parser = SSEParser()

    var onEvent: ((SSEEvent) -> Void)?
    var onFinalResponse: ((TriageFinalResponse) -> Void)?
    var onError: ((String) -> Void)?
    var onComplete: (() -> Void)?

    // MARK: - Health check

    func checkHealth() async -> Bool {
        guard let url = URL(string: "\(Self.baseURL)/health") else { return false }
        do {
            let (_, response) = try await session.data(from: url)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    // MARK: - Triage stream

    func stream(request triageRequest: TriageRequest) async {
        guard let url = URL(string: "\(Self.baseURL)/v1/triage/stream") else {
            onError?("Invalid API URL")
            return
        }

        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.timeoutInterval = 300

        do {
            urlRequest.httpBody = try JSONEncoder().encode(triageRequest)
        } catch {
            onError?("Failed to encode request: \(error.localizedDescription)")
            return
        }

        // Wire up parser — SSEParser is nonisolated, so we capture self weakly
        // and dispatch events back to MainActor
        let onEventHandler = self.onEvent
        let onErrorHandler = self.onError
        let onFinalHandler = self.onFinalResponse

        parser.onEvent = { @Sendable [weak self] name, rawJSON in
            let event = SSEEvent(name: name, timestamp: Date(), rawJSON: rawJSON)

            Task { @MainActor [weak self] in
                onEventHandler?(event)

                if name == "triage.final" {
                    self?.decodeFinal(rawJSON: rawJSON, handler: onFinalHandler)
                }
                if name == "triage.error" {
                    // Try to extract the error message
                    if let data = rawJSON.data(using: .utf8),
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let errMsg = json["error"] as? String {
                        onErrorHandler?(errMsg)
                    } else {
                        onErrorHandler?("Unknown pipeline error")
                    }
                }
            }
        }

        do {
            let (asyncBytes, response) = try await session.bytes(for: urlRequest)

            guard let httpResponse = response as? HTTPURLResponse else {
                onError?("Invalid HTTP response")
                return
            }
            guard httpResponse.statusCode == 200 else {
                // Collect the error body for non-200 responses (e.g. 422 validation errors)
                var errorBody = ""
                for try await byte in asyncBytes {
                    errorBody += String(bytes: [byte], encoding: .utf8) ?? ""
                }
                let detail = Self.parseErrorDetail(errorBody)
                onError?("HTTP \(httpResponse.statusCode): \(detail)")
                return
            }

            var accumulator = ""
            for try await byte in asyncBytes {
                accumulator += String(bytes: [byte], encoding: .utf8) ?? ""
                if accumulator.count > 64 {
                    parser.feed(accumulator)
                    accumulator = ""
                }
            }
            if !accumulator.isEmpty { parser.feed(accumulator) }
            parser.flush()
            onComplete?()

        } catch {
            onError?(error.localizedDescription)
        }
    }

    // MARK: - Decode final payload

    private func decodeFinal(rawJSON: String, handler: ((TriageFinalResponse) -> Void)?) {
        guard let data = rawJSON.data(using: .utf8),
              let finalResponse = try? JSONDecoder().decode(TriageFinalResponse.self, from: data) else {
            return
        }
        handler?(finalResponse)
    }

    // MARK: - Parse error body (e.g. Pydantic 422 validation errors)

    private static func parseErrorDetail(_ body: String) -> String {
        guard let data = body.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let details = json["detail"] as? [[String: Any]] else {
            return body.prefix(200).isEmpty ? "Unknown error" : String(body.prefix(200))
        }
        let messages = details.compactMap { item -> String? in
            guard let msg = item["msg"] as? String,
                  let loc = item["loc"] as? [Any] else { return nil }
            let field = loc.compactMap { "\($0)" }.joined(separator: ".")
            return "\(field): \(msg)"
        }
        return messages.isEmpty ? body.prefix(200).description : messages.joined(separator: "; ")
    }
}
