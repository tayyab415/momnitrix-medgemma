import Foundation
import HealthKit
import Combine

@MainActor
final class HealthReader: ObservableObject {

    private let store = HKHealthStore()

    private let readTypes: Set<HKSampleType> = {
        var types = Set<HKSampleType>()
        types.insert(HKQuantityType(.heartRate))
        types.insert(HKQuantityType(.heartRateVariabilitySDNN))
        types.insert(HKQuantityType(.oxygenSaturation))
        types.insert(HKQuantityType(.bodyTemperature))
        types.insert(HKQuantityType(.appleSleepingWristTemperature))
        return types
    }()

    @Published var heartRate: Double?
    @Published var hrv: Double?
    @Published var spo2: Double?
    @Published var wristTempC: Double?
    @Published var permissionGranted = false
    @Published var permissionError: String?
    @Published var debugLog: String = ""

    // MARK: - Permission

    func requestPermission() async {
        guard HKHealthStore.isHealthDataAvailable() else {
            permissionError = "HealthKit not available on this device."
            log("❌ HealthKit not available")
            return
        }
        do {
            // On watchOS, requestAuthorization completes without error even if
            // user denies read access (Apple privacy). We mark granted and try to read.
            try await store.requestAuthorization(toShare: [], read: Set(readTypes.map { $0 as HKObjectType }))
            permissionGranted = true
            log("✅ Authorization requested (no error)")
            await readAll()
        } catch {
            permissionError = error.localizedDescription
            log("❌ Authorization error: \(error.localizedDescription)")
        }
    }

    // MARK: - Read latest sample for all types

    func readAll() async {
        log("🔍 Reading all vitals…")

        async let hrResult = readLatest(
            type: HKQuantityType(.heartRate),
            unit: HKUnit(from: "count/min"),
            label: "Heart Rate"
        )
        async let hrvResult = readLatest(
            type: HKQuantityType(.heartRateVariabilitySDNN),
            unit: .secondUnit(with: .milli),
            label: "HRV"
        )
        async let o2Result = readLatest(
            type: HKQuantityType(.oxygenSaturation),
            unit: .percent(),
            label: "SpO2"
        )
        async let tempResult = readLatestTemp()

        heartRate  = await hrResult
        hrv        = await hrvResult
        spo2       = await o2Result
        wristTempC = await tempResult

        log("📊 HR=\(heartRate.map { "\($0)" } ?? "nil"), HRV=\(hrv.map { "\($0)" } ?? "nil"), SpO2=\(spo2.map { "\($0)" } ?? "nil"), Temp=\(wristTempC.map { "\($0)" } ?? "nil")")
    }

    private func readLatest(type: HKQuantityType, unit: HKUnit, label: String) async -> Double? {
        return await withCheckedContinuation { continuation in
            let sort = NSSortDescriptor(
                key: HKSampleSortIdentifierEndDate,
                ascending: false
            )
            // Query samples from the last 24 hours
            let now = Date()
            let oneDayAgo = now.addingTimeInterval(-86400)
            let predicate = HKQuery.predicateForSamples(withStart: oneDayAgo, end: now, options: .strictStartDate)

            let query = HKSampleQuery(
                sampleType: type,
                predicate: predicate,
                limit: 1,
                sortDescriptors: [sort]
            ) { _, samples, error in
                if let error {
                    print("[HealthReader] ⚠️ \(label) query error: \(error.localizedDescription)")
                }
                guard let sample = samples?.first as? HKQuantitySample else {
                    print("[HealthReader] ℹ️ \(label): no samples found (count: \(samples?.count ?? 0))")
                    continuation.resume(returning: nil)
                    return
                }
                let value = sample.quantity.doubleValue(for: unit)
                print("[HealthReader] ✅ \(label) = \(value) (from \(sample.endDate))")
                continuation.resume(returning: value)
            }
            store.execute(query)
        }
    }

    private func readLatestTemp() async -> Double? {
        if let t = await readLatest(
            type: HKQuantityType(.bodyTemperature),
            unit: .degreeCelsius(),
            label: "Body Temp"
        ) {
            return t
        }
        if let delta = await readLatest(
            type: HKQuantityType(.appleSleepingWristTemperature),
            unit: .degreeCelsius(),
            label: "Wrist Temp"
        ) {
            return 36.5 + delta
        }
        return nil
    }

    // MARK: - Debug log

    private func log(_ msg: String) {
        let ts = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        debugLog += "[\(ts)] \(msg)\n"
        print("[HealthReader] \(msg)")
    }

    // MARK: - Formatted display helpers

    var heartRateDisplay: String {
        heartRate.map { "\(Int($0)) bpm" } ?? "—"
    }
    var hrvDisplay: String {
        hrv.map { "\(Int($0)) ms" } ?? "—"
    }
    var spo2Display: String {
        spo2.map { String(format: "%.0f%%", $0 * 100) } ?? "—"
    }
    var tempDisplay: String {
        wristTempC.map { String(format: "%.1f °C", $0) } ?? "—"
    }

    var spo2Percent: Double {
        (spo2 ?? 0) * 100.0
    }
}
