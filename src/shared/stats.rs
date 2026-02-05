/// Statistics about the planning process.
#[derive(Default, Clone)]
pub(crate) struct PlannerStats {
    /// Number of coordinate array reads performed.
    pub(crate) coord_reads: u64,
}
