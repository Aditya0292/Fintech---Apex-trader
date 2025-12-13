
import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import path from 'path';

export const dynamic = 'force-dynamic'; // No caching

export async function GET(request: Request) {
    const { searchParams } = new URL(request.url);
    const assets = searchParams.get('assets') || 'all';

    // Path to python script relative to web-dashboard
    // web-dashboard is at root/web-dashboard
    // tools is at root/tools
    const scriptPath = path.resolve(process.cwd(), '..', 'tools', 'predict_all.py');

    // Command: python tools/predict_all.py --assets all --json
    const cmd = `python "${scriptPath}" --assets ${assets} --json`;

    console.log(`Executing: ${cmd}`);

    return new Promise((resolve) => {
        exec(cmd, { cwd: path.resolve(process.cwd(), '..') }, (error, stdout, stderr) => {
            if (error) {
                console.error(`Exec Error: ${error}`);
                console.error(`Stderr: ${stderr}`);
                resolve(NextResponse.json({ error: "Analysis Failed", details: stderr }, { status: 500 }));
                return;
            }

            try {
                // Find the JSON part. Sometimes script prints logs too.
                // Assumption: The script outputs valid JSON as the LAST line or mostly only JSON if --json is passed.
                // My refactor ensured it prints JSON.
                const data = JSON.parse(stdout);
                resolve(NextResponse.json(data));
            } catch (e) {
                console.error("JSON Parse Error", e);
                console.log("Raw Output:", stdout);
                resolve(NextResponse.json({ error: "Invalid JSON from Backend", raw: stdout }, { status: 500 }));
            }
        });
    });
}
