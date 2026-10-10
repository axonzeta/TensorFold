//! ngram-learn: what a served prompt forward leaves for training (Forward.capture; null in serving). Every layer's
//! input as the layer saw it (the previous layer's write-back applied, before the n-gram block), each layer's expert
//! picks, and the final streams: the backward then recomputes each layer from the served activations and routes.
pub const Capture = struct {
    /// bf16 [layers][rows][streams * hidden]
    h_in: u64,
    /// i32 [layers][rows][slots] (top-k routed, then the shared expert's slot)
    picks: u64,
    /// bf16 [rows][streams * hidden]
    final: u64,
    rows: usize,
    slots: usize,
};
