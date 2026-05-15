"""
Technical corpus: 10 paragraphs covering distributed systems topics.

Chunk IDs 0-9 correspond to FAISS index positions in insertion order.
Ground-truth relevance judgements are defined in BENCHMARK_QUERIES.
"""

CHUNKS: list[str] = [
    # 0 — load balancing / peak traffic
    (
        "Load balancing distributes incoming requests across multiple server instances "
        "to prevent any single node from becoming a bottleneck. During peak traffic, "
        "the load balancer applies algorithms such as round-robin, least-connections, "
        "or weighted routing to spread the workload. Health checks continuously probe "
        "backends and automatically route traffic away from unhealthy instances, "
        "ensuring high availability under sudden demand spikes."
    ),
    # 1 — horizontal scaling / autoscaling
    (
        "Horizontal scaling adds new server instances in response to increased traffic "
        "rather than vertically upgrading a single machine. Cloud platforms support "
        "autoscaling policies that monitor CPU utilisation, request queue depth, or "
        "custom metrics and provision new instances within minutes. This elastic "
        "capacity management prevents service degradation during peak load periods "
        "without over-provisioning resources during normal operation."
    ),
    # 2 — in-memory caching
    (
        "In-memory caching with systems such as Redis or Memcached stores frequently "
        "accessed data in RAM, reducing the number of expensive database round-trips. "
        "Cache entries are assigned a time-to-live (TTL) that balances data freshness "
        "against hit rate. Write-through and write-behind strategies determine when "
        "cached data is synchronised back to the primary datastore, preventing stale "
        "reads without sacrificing write throughput."
    ),
    # 3 — circuit breakers and fault tolerance
    (
        "The circuit-breaker pattern prevents cascading failures in distributed systems "
        "by temporarily halting requests to an unresponsive downstream service. When "
        "the error rate crosses a threshold, the breaker opens and returns a fallback "
        "response immediately, giving the failing service time to recover. After a "
        "configurable cool-down window the breaker moves to a half-open state and "
        "allows a probe request through to test service recovery."
    ),
    # 4 — database connection pooling
    (
        "Database connection pooling maintains a pool of pre-established connections "
        "that incoming requests can borrow rather than creating a new connection per "
        "query. Connection creation is expensive — it involves TCP handshake, "
        "authentication, and session initialisation — so pooling dramatically reduces "
        "per-query latency and protects the database from connection exhaustion during "
        "traffic spikes. Pool size is tuned based on DB server thread limits and "
        "expected concurrent query volume."
    ),
    # 5 — rate limiting
    (
        "Rate limiting enforces an upper bound on how many requests a client or "
        "service can make within a time window, protecting backend resources from "
        "overload. Token-bucket and sliding-window algorithms are common "
        "implementations; the former allows short bursts while the latter provides "
        "a smoother request distribution. When a limit is exceeded, the service "
        "returns HTTP 429 Too Many Requests, signalling clients to back off and retry "
        "with exponential backoff."
    ),
    # 6 — async processing / message queues
    (
        "Asynchronous processing decouples producers from consumers by introducing "
        "a message queue such as Kafka or Cloud Pub/Sub between them. Long-running "
        "tasks — image processing, report generation, email dispatch — are offloaded "
        "to background workers that consume messages at their own pace, keeping API "
        "response times low. Dead-letter queues capture messages that fail after "
        "repeated retries, preventing data loss while isolating failure modes."
    ),
    # 7 — CDN / edge caching
    (
        "Content delivery networks cache static and semi-static assets at edge nodes "
        "geographically close to end users, reducing origin-server load and improving "
        "perceived latency. CDN cache policies are controlled via HTTP Cache-Control "
        "headers; stale-while-revalidate allows serving cached content while "
        "asynchronously refreshing it in the background. Dynamic content can be "
        "accelerated using CDN edge compute functions that execute business logic "
        "without round-tripping to the origin."
    ),
    # 8 — memory management under pressure
    (
        "Under memory pressure, the operating system may resort to swapping pages to "
        "disk, causing latency spikes of several orders of magnitude. Applications "
        "mitigate this by pre-allocating buffers, tuning garbage collector heap sizes, "
        "and using off-heap or memory-mapped file storage for large datasets. Container "
        "resource limits enforce a hard ceiling on memory consumption, preventing a "
        "runaway process from degrading co-located services on the same node."
    ),
    # 9 — monitoring and alerting
    (
        "Comprehensive monitoring captures system health through metrics (CPU, memory, "
        "request rate, error rate), distributed traces that track request flows across "
        "services, and structured logs. SLO-based alerting fires when a rolling-window "
        "error budget is being consumed faster than expected, giving teams early warning "
        "before an SLA breach. Dashboards visualise P50, P95, and P99 latency "
        "percentiles to surface tail-latency regressions that averages would mask."
    ),
]


BENCHMARK_QUERIES: list[dict] = [
    {
        "query": "How does the system handle peak load?",
        "relevant_ids": [0, 1, 5],
        "notes": (
            "Covers load balancing (0), autoscaling (1), and rate limiting (5) — "
            "the three primary mechanisms for absorbing traffic surges."
        ),
    },
    {
        "query": "What caching strategies improve latency under high read volume?",
        "relevant_ids": [2, 7, 3],
        "notes": (
            "In-memory caching (2) and CDN edge caching (7) directly address "
            "read latency; circuit breakers (3) protect cache-misses from cascading."
        ),
    },
    {
        "query": "How is database performance maintained during traffic spikes?",
        "relevant_ids": [4, 0, 9],
        "notes": (
            "Connection pooling (4) is the primary DB-specific mechanism; "
            "load balancing (0) prevents DB overload from upstream; "
            "monitoring (9) detects regressions in query latency."
        ),
    },
    {
        "query": "How does the system prevent cascading failures under partial outages?",
        "relevant_ids": [3, 6, 5],
        "notes": (
            "Circuit breakers (3) isolate failures; async queues (6) decouple "
            "producers from failing consumers; rate limiting (5) prevents overload "
            "amplification during retry storms."
        ),
    },
]
