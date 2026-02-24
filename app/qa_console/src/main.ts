/**
 * Momnitrix QA Console – Entry Point
 *
 * Boots the app by wiring DOM events and setting initial state.
 * This file is the esbuild entry: it bundles into `app.js`.
 */

import { wireEvents, showResultCard, resetToDefaults } from './ui/dom';

wireEvents();
showResultCard(0);
resetToDefaults();
