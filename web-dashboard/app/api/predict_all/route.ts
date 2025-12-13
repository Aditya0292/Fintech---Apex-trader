import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import path from 'path';
import fs from 'fs';
import { promisify } from 'util';

const execAsync = promisify(exec);
export const dynamic = 'force-dynamic'; // No caching

const CACHE_FILE = path.resolve(process.cwd(), '..', 'data', 'prediction_cache.json');
const CACHE_DURATION_MS = 5 * 60 * 1000; // 5 Minutes

async function checkAndRebuildScalers() {
    const dataDir = path.resolve(process.cwd(), '..', 'data');
    const scalerPath = path.join(dataDir, 'scaler_features_XAUUSD_1h.pkl'); // Check one key file

    // Simple check: if main scaler missing, assume all missing/corrupt
    if (!fs.existsSync(scalerPath)) {
        console.log("⚠️ Scalers missing. Triggering Auto-Recovery (rebuild_scalers.py)...");
        const scriptPath = path.resolve(process.cwd(), '..', 'tools', 'rebuild_scalers.py');
        try {
            await execAsync(`python "${scriptPath}"`, { cwd: path.resolve(process.cwd(), '..') });
            console.log("✅ Auto-Recovery Complete.");
        } catch (error) {
            console.error("❌ Auto-Recovery Failed:", error);
            // Proceed anyway, predict_all might fail but we tried
        }
    }
}

export async function GET(request: Request) {
    const { searchParams } = new URL(request.url);
    const assets = searchParams.get('assets') || 'all';
    const forceRefresh = searchParams.get('refresh') === 'true';

    // 1. AUTO-RECOVERY CHECK
    await checkAndRebuildScalers();

    // 2. CACHE CHECK (Skip if forceRefresh is true)
    if (!forceRefresh && fs.existsSync(CACHE_FILE)) {
        try {
            const stats = fs.statSync(CACHE_FILE);
            const age = Date.now() - stats.mtimeMs;

            if (age < CACHE_DURATION_MS) {
                console.log(`⚡ Serving from Cache (${(age / 1000).toFixed(0)}s old)`);
                const cachedData = fs.readFileSync(CACHE_FILE, 'utf-8');
                return NextResponse.json(JSON.parse(cachedData));
            }
        } catch (e) {
            console.warn("Cache read error, ignoring:", e);
        }
    }

    // 3. RUN PREDICTION
    const scriptPath = path.resolve(process.cwd(), '..', 'tools', 'predict_all.py');
    const cmd = `python "${scriptPath}" --assets ${assets} --json`;

    console.log(`🚀 Executing: ${cmd}`);

    return new Promise((resolve) => {
        exec(cmd, { cwd: path.resolve(process.cwd(), '..') }, (error, stdout, stderr) => {
            if (error) {
                console.error(`Exec Error: ${error}`);
                console.error(`Stderr: ${stderr}`);
                resolve(NextResponse.json({ error: "Analysis Failed", details: stderr }, { status: 500 }));
                return;
            }

            try {
                // Parse Data
                const data = JSON.parse(stdout);

                // 4. SAVE TO CACHE
                try {
                    fs.writeFileSync(CACHE_FILE, JSON.stringify(data));
                    console.log("💾 Cache updated.");
                } catch (cacheErr) {
                    console.error("Failed to write cache:", cacheErr);
                }

                resolve(NextResponse.json(data));
            } catch (e) {
                console.error("JSON Parse Error", e);
                console.log("Raw Output:", stdout);
                resolve(NextResponse.json({ error: "Invalid JSON from Backend", raw: stdout }, { status: 500 }));
            }
        });
    });
}
