<?php

use App\Http\Controllers\PredictionController;
use Illuminate\Support\Facades\Route;

Route::get('/', [PredictionController::class, 'index'])->name('predictions.index');
Route::post('/predict', [PredictionController::class, 'store'])->name('predictions.store');
Route::get('/history', [PredictionController::class, 'history'])->name('predictions.history');
