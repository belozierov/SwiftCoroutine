@usableFromInline
internal final class SharedCoroutineDispatcher: CoroutineTaskExecutor {
    
    private let stackSize, capacity: Int
    private var lifo = LifoQueue<SharedCoroutineQueue>()
    private var fifo = FifoQueue<SharedCoroutineQueue>()
    private var queuesCount = 0
    
    internal init(capacity: Int, stackSize: Coroutine.StackSize) {
        self.stackSize = stackSize.size
        self.capacity = capacity
    }
    
    @usableFromInline
    internal func execute(on scheduler: CoroutineScheduler, task: @escaping () -> Void) {
        scheduler.scheduleTask {
            self.getFreeQueue().start(dispatcher: self, scheduler: scheduler, task: task)
        }
    }
    
    private func getFreeQueue() -> SharedCoroutineQueue {
        while let queue = lifo.pop() ?? fifo.pop() {
            atomicAdd(&queuesCount, value: -1)
            queue.inQueue = false
            if queue.occupy() { return queue }
        }
        return SharedCoroutineQueue(stackSize: stackSize)
    }
    
    internal func push(_ queue: SharedCoroutineQueue) {
        if queue.started != 0 {
            if queue.inQueue { return }
            queue.inQueue = true
            fifo.push(queue)
            atomicAdd(&queuesCount, value: 1)
        } else if queuesCount < capacity {
            lifo.push(queue)
            atomicAdd(&queuesCount, value: 1)
        }
    }
    
    deinit {
        lifo.free()
        fifo.free()
    }
    
}
