@extends('layouts.app')

@section('title', 'Scan History — SickleVision')

@section('content')

<section class="hero" style="padding-top:2.5rem;padding-bottom:1.5rem;">
  <div class="container">
    <p class="hero-label">MySQL-Backed Log</p>
    <h1>Scan <em>History</em></h1>
    <p>Every prediction made through SickleVision is logged here — no images are stored, only the result.</p>
  </div>
</section>

<main class="container pb-5" style="position:relative;z-index:1;">

  <div class="stats-strip">
    <div class="stat-box">
      <div class="stat-label">Total Scans</div>
      <div class="stat-value">{{ number_format($stats['total']) }}</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Sickle-Positive</div>
      <div class="stat-value">{{ number_format($stats['sickle']) }}</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Positive Rate</div>
      <div class="stat-value">{{ $stats['sickle_pct'] }}%</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Avg. Confidence</div>
      <div class="stat-value">{{ $stats['avg_confidence'] }}%</div>
    </div>
  </div>

  <div class="app-card p-3 p-md-4">
    <div class="section-label">All Scans</div>

    @if ($predictions->isEmpty())
      <div class="d-flex flex-column align-items-center justify-content-center text-center py-5" style="gap:.7rem;">
        <i class="bi bi-inbox" style="font-size:2.6rem;color:var(--border);"></i>
        <p style="color:var(--muted);font-family:var(--mono);font-size:.75rem;margin:0;">
          No scans yet. <a href="{{ route('predictions.index') }}" style="color:var(--accent2);">Run one &rarr;</a>
        </p>
      </div>
    @else
      <div style="overflow-x:auto;">
        <table class="history-table">
          <thead>
            <tr><th>#</th><th>File</th><th>Result</th><th>Confidence</th><th>Raw Score</th><th>When</th></tr>
          </thead>
          <tbody>
            @foreach ($predictions as $p)
            <tr>
              <td style="color:var(--muted);">{{ $p->id }}</td>
              <td>{{ \Illuminate\Support\Str::limit($p->original_filename, 32) }}</td>
              <td>
                <span class="result-badge {{ $p->isSickleCell() ? 'sickle' : 'normal' }}" style="font-size:.75rem;padding:.3rem .7rem;">
                  {{ $p->predicted_class }}
                </span>
              </td>
              <td>{{ number_format($p->confidence * 100, 1) }}%</td>
              <td style="color:var(--muted);">{{ number_format($p->raw_score, 6) }}</td>
              <td style="color:var(--muted);">{{ $p->created_at->format('Y-m-d H:i') }}</td>
            </tr>
            @endforeach
          </tbody>
        </table>
      </div>

      <div class="mt-3 d-flex justify-content-center">
        {{ $predictions->links() }}
      </div>
    @endif
  </div>

</main>

@endsection

@section('extra-style')
.pagination { justify-content: center; }
.page-link {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
  font-family: var(--mono);
  font-size: .8rem;
}
.page-item.active .page-link { background: var(--accent) !important; border-color: var(--accent) !important; color: #fff !important; }
.page-item.disabled .page-link { color: var(--muted) !important; }
@endsection
