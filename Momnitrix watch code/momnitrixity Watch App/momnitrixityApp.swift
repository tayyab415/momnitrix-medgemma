//
//  momnitrixityApp.swift
//  momnitrixity Watch App
//
//  Created by Sahil Tanna on 24/02/26.
//

import SwiftUI

@main
struct momnitrixity_Watch_AppApp: App {

    @State private var viewModel = TriageViewModel()

    init() {
        // Activate WatchConnectivity for iPhone image transfer
        WatchSessionManager.shared.activate()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(viewModel)
        }
    }
}

struct RootView: View {
    @Environment(TriageViewModel.self) var vm

    var body: some View {
        switch vm.screen {
        case .home:
            HomeView()
        case .inputSheet:
            InputSheetView()
        case .diagnosing:
            DiagnosisView()
        case .results:
            ResultsView()
        }
    }
}
