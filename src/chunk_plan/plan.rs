use super::types::DimChunkRange;

#[derive(Debug, Clone)]
pub(crate) enum ChunkPlanNode {
    Empty,
    AllChunks,
    Rect(Vec<DimChunkRange>),
    Explicit(Vec<Vec<u64>>),
    Union(Vec<ChunkPlanNode>),
}

impl ChunkPlanNode {
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            ChunkPlanNode::Empty => true,
            ChunkPlanNode::Explicit(v) => v.is_empty(),
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ChunkPlan {
    grid_shape: Vec<u64>,
    root: ChunkPlanNode,
}

impl ChunkPlan {
    pub(crate) fn all(grid_shape: Vec<u64>) -> Self {
        Self {
            grid_shape,
            root: ChunkPlanNode::AllChunks,
        }
    }

    pub(crate) fn from_root(grid_shape: Vec<u64>, root: ChunkPlanNode) -> Self {
        Self {
            grid_shape,
            root,
        }
    }

    pub(crate) fn grid_shape(&self) -> &[u64] {
        &self.grid_shape
    }

    pub(crate) fn into_index_iter(self) -> ChunkIndexIter {
        ChunkIndexIter::new(self.root, self.grid_shape)
    }
}

pub(crate) struct ChunkIndexIter {
    stack: Vec<OwnedIterFrame>,
}

enum OwnedIterFrame {
    Empty,
    AllChunks(AllChunksIter),
    Rect(RectIter),
    Explicit(ExplicitIter),
    Union(UnionOwnedIter),
}

impl ChunkIndexIter {
    fn new(root: ChunkPlanNode, grid_shape: Vec<u64>) -> Self {
        let mut it = Self { stack: Vec::new() };
        it.push_node(root, grid_shape);
        it
    }

    fn push_node(&mut self, node: ChunkPlanNode, grid_shape: Vec<u64>) {
        match node {
            ChunkPlanNode::Empty => self.stack.push(OwnedIterFrame::Empty),
            ChunkPlanNode::AllChunks => self
                .stack
                .push(OwnedIterFrame::AllChunks(AllChunksIter::new(&grid_shape))),
            ChunkPlanNode::Rect(ranges) => self
                .stack
                .push(OwnedIterFrame::Rect(RectIter::new(&ranges))),
            ChunkPlanNode::Explicit(items) => self
                .stack
                .push(OwnedIterFrame::Explicit(ExplicitIter::new(items))),
            ChunkPlanNode::Union(children) => self.stack.push(OwnedIterFrame::Union(
                UnionOwnedIter::new(children, grid_shape),
            )),
        }
    }
}

impl Iterator for ChunkIndexIter {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let frame = self.stack.pop()?;
            match frame {
                OwnedIterFrame::Empty => continue,
                OwnedIterFrame::AllChunks(mut it) => {
                    if let Some(v) = it.next() {
                        self.stack.push(OwnedIterFrame::AllChunks(it));
                        return Some(v);
                    }
                }
                OwnedIterFrame::Rect(mut it) => {
                    if let Some(v) = it.next() {
                        self.stack.push(OwnedIterFrame::Rect(it));
                        return Some(v);
                    }
                }
                OwnedIterFrame::Explicit(mut it) => {
                    if let Some(v) = it.next() {
                        self.stack.push(OwnedIterFrame::Explicit(it));
                        return Some(v);
                    }
                }
                OwnedIterFrame::Union(mut it) => {
                    if let Some(v) = it.next() {
                        self.stack.push(OwnedIterFrame::Union(it));
                        return Some(v);
                    }
                }
            }
        }
    }
}

struct AllChunksIter {
    grid_shape: Vec<u64>,
    cur: Vec<u64>,
    started: bool,
    done: bool,
}

impl AllChunksIter {
    fn new(grid_shape: &[u64]) -> Self {
        let cur = vec![0; grid_shape.len()];
        Self {
            grid_shape: grid_shape.to_vec(),
            cur,
            started: false,
            done: grid_shape.is_empty(),
        }
    }
}

impl Iterator for AllChunksIter {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if !self.started {
            self.started = true;
            return Some(self.cur.clone());
        }
        // advance lexicographically (last dim fastest)
        for i in (0..self.cur.len()).rev() {
            self.cur[i] += 1;
            if self.cur[i] < self.grid_shape[i] {
                return Some(self.cur.clone());
            }
            self.cur[i] = 0;
        }
        self.done = true;
        None
    }
}

struct RectIter {
    ranges: Vec<DimChunkRange>,
    cur: Vec<u64>,
    started: bool,
    done: bool,
}

impl RectIter {
    fn new(ranges: &[DimChunkRange]) -> Self {
        let cur = ranges.iter().map(|r| r.start_chunk).collect::<Vec<_>>();
        let done =
            ranges.is_empty() || ranges.iter().any(|r| r.end_chunk_inclusive < r.start_chunk);
        Self {
            ranges: ranges.to_vec(),
            cur,
            started: false,
            done,
        }
    }
}

impl Iterator for RectIter {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if !self.started {
            self.started = true;
            return Some(self.cur.clone());
        }
        for i in (0..self.cur.len()).rev() {
            self.cur[i] += 1;
            if self.cur[i] <= self.ranges[i].end_chunk_inclusive {
                return Some(self.cur.clone());
            }
            self.cur[i] = self.ranges[i].start_chunk;
        }
        self.done = true;
        None
    }
}

struct ExplicitIter {
    items: Vec<Vec<u64>>,
    idx: usize,
}

impl ExplicitIter {
    fn new(items: Vec<Vec<u64>>) -> Self {
        Self { items, idx: 0 }
    }
}

impl Iterator for ExplicitIter {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.items.len() {
            return None;
        }
        let v = self.items[self.idx].clone();
        self.idx += 1;
        Some(v)
    }
}

struct UnionOwnedIter {
    children: Vec<ChunkPlanNode>,
    child_idx: usize,
    child_iter: Option<ChunkIndexIter>,
    grid_shape: Vec<u64>,
}

impl UnionOwnedIter {
    fn new(children: Vec<ChunkPlanNode>, grid_shape: Vec<u64>) -> Self {
        Self {
            children,
            child_idx: 0,
            child_iter: None,
            grid_shape,
        }
    }
}

impl Iterator for UnionOwnedIter {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(it) = &mut self.child_iter {
                if let Some(v) = it.next() {
                    return Some(v);
                }
            }
            if self.child_idx >= self.children.len() {
                return None;
            }
            let node = self.children[self.child_idx].clone();
            self.child_iter = Some(ChunkIndexIter::new(node, self.grid_shape.clone()));
            self.child_idx += 1;
        }
    }
}

