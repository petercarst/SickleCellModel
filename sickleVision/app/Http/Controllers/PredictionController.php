<?php

namespace App\Http\Controllers;

use App\Models\Prediction;
use Illuminate\Http\Client\ConnectionException;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\View\View;

class PredictionController extends Controller
{
    public function index(): View
    {
        return view('predictions.index', [
            'stats' => $this->stats(),
            'recent' => Prediction::latest()->take(5)->get(),
        ]);
    }

    public function history(): View
    {
        return view('predictions.history', [
            'predictions' => Prediction::latest()->paginate(15),
            'stats' => $this->stats(),
        ]);
    }

    public function store(Request $request): JsonResponse
    {
        $request->validate([
            'file' => ['required', 'file', 'mimes:jpeg,jpg,png,webp', 'max:10240'],
        ]);

        $file = $request->file('file');

        try {
            $response = Http::timeout(30)
                ->attach('file', file_get_contents($file->getRealPath()), $file->getClientOriginalName())
                ->post(config('services.fastapi.url'));
        } catch (ConnectionException) {
            return response()->json([
                'error' => 'Could not reach the prediction service. Please ensure the FastAPI server is running.',
            ], 503);
        }

        if ($response->failed()) {
            return response()->json([
                'error' => $response->json('detail') ?? 'TF Serving / FastAPI error.',
            ], $response->status());
        }

        $data = $response->json();

        $prediction = Prediction::create([
            'original_filename' => $file->getClientOriginalName(),
            'predicted_class' => $data['class'],
            'confidence' => $data['confidence'],
            'raw_score' => $data['raw_score'] ?? $data['confidence'],
            'ip_address' => $request->ip(),
        ]);

        return response()->json([
            'class' => $prediction->predicted_class,
            'confidence' => $prediction->confidence,
            'raw_score' => $prediction->raw_score,
        ]);
    }

    private function stats(): array
    {
        $total = Prediction::count();
        $sickle = Prediction::where('predicted_class', 'Sickle Cell')->count();

        return [
            'total' => $total,
            'sickle' => $sickle,
            'sickle_pct' => $total > 0 ? round($sickle / $total * 100, 1) : 0,
            'avg_confidence' => $total > 0 ? round(Prediction::avg('confidence') * 100, 1) : 0,
        ];
    }
}
