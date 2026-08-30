<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Prediction extends Model
{
    protected $fillable = [
        'original_filename',
        'predicted_class',
        'confidence',
        'raw_score',
        'ip_address',
    ];

    protected $casts = [
        'confidence' => 'float',
        'raw_score' => 'float',
    ];

    public function isSickleCell(): bool
    {
        return $this->predicted_class === 'Sickle Cell';
    }
}
