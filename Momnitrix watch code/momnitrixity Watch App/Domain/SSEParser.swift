import Foundation

// SSE text/event-stream parser.
// Needs to be nonisolated (not @MainActor) since it processes data from URLSession bytes.

nonisolated final class SSEParser: @unchecked Sendable {

    private var buffer = ""
    var onEvent: (@Sendable (String, String) -> Void)?  // (eventName, rawJSONString)

    func feed(_ chunk: String) {
        buffer += chunk.replacingOccurrences(of: "\r", with: "")
        while let range = buffer.range(of: "\n\n") {
            let block = String(buffer[buffer.startIndex..<range.lowerBound])
            buffer = String(buffer[range.upperBound...])
            parseBlock(block)
        }
    }

    func flush() {
        if !buffer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            parseBlock(buffer)
        }
        buffer = ""
    }

    private func parseBlock(_ block: String) {
        var eventName = "message"
        var dataLines: [String] = []

        for line in block.split(separator: "\n", omittingEmptySubsequences: false) {
            let lineStr = String(line)
            if lineStr.isEmpty || lineStr.hasPrefix(":") { continue }
            if lineStr.hasPrefix("event:") {
                eventName = lineStr.dropFirst(6).trimmingCharacters(in: .whitespaces)
            } else if lineStr.hasPrefix("data:") {
                dataLines.append(lineStr.dropFirst(5).trimmingCharacters(in: .whitespaces))
            }
        }

        let raw = dataLines.joined(separator: "\n")
        guard !raw.isEmpty else { return }

        onEvent?(eventName, raw)
    }
}
