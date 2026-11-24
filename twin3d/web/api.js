/**
 * api.js — Client for FastAPI twin3d endpoints with bounded snapshot prefetch cache.
 */

export class TwinAPI {
  constructor() {
    this.snapshotCache = new Map();
    this.maxCache = 24;
  }

  async _json(url, options = undefined) {
    const resp = await fetch(url, options);
    let body = null;
    try {
      body = await resp.json();
    } catch {
      body = null;
    }
    if (!resp.ok) {
      const msg = body?.detail || `HTTP ${resp.status}`;
      const err = new Error(msg);
      err.status = resp.status;
      throw err;
    }
    return body;
  }

  getHealth(cluster = null) {
    const q = cluster ? `?cluster=${encodeURIComponent(cluster)}` : "";
    return this._json(`/api/health${q}`);
  }

  getMeta() {
    return this._json("/api/meta");
  }

  async getSnapshot(cluster, tIso, horizon) {
    const key = `${cluster}|${tIso}|${horizon}`;
    if (this.snapshotCache.has(key)) {
      return this.snapshotCache.get(key);
    }
    const data = await this._json(
      `/api/snapshot?cluster=${encodeURIComponent(cluster)}&t=${encodeURIComponent(tIso)}&horizon=${horizon}`
    );
    this._putCache(key, data);
    return data;
  }

  prefetchSnapshots(cluster, times, horizon) {
    for (const tIso of times) {
      if (!tIso) continue;
      const key = `${cluster}|${tIso}|${horizon}`;
      if (!this.snapshotCache.has(key)) {
        this._json(
          `/api/snapshot?cluster=${encodeURIComponent(cluster)}&t=${encodeURIComponent(tIso)}&horizon=${horizon}`
        )
          .then((d) => this._putCache(key, d))
          .catch(() => {});
      }
    }
  }

  _putCache(key, val) {
    if (this.snapshotCache.has(key)) {
      this.snapshotCache.delete(key);
    }
    this.snapshotCache.set(key, val);
    while (this.snapshotCache.size > this.maxCache) {
      const first = this.snapshotCache.keys().next().value;
      this.snapshotCache.delete(first);
    }
  }

  getCompare(cluster, tIso, horizon) {
    return this._json(
      `/api/compare?cluster=${encodeURIComponent(cluster)}&t=${encodeURIComponent(tIso)}&horizon=${horizon}`
    );
  }

  getSeries(cluster, host, startIso, endIso, horizon) {
    let url = `/api/series?cluster=${encodeURIComponent(cluster)}&host=${encodeURIComponent(host)}&horizon=${horizon}`;
    if (startIso) url += `&start=${encodeURIComponent(startIso)}`;
    if (endIso) url += `&end=${encodeURIComponent(endIso)}`;
    return this._json(url);
  }

  getAlerts(cluster, tIso) {
    return this._json(
      `/api/alerts?cluster=${encodeURIComponent(cluster)}&t=${encodeURIComponent(tIso)}`
    );
  }

  postWhatIf(payload) {
    return this._json("/api/whatif", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }
}
