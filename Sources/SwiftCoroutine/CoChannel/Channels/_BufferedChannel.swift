//
//  _BufferedChannel.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 07.06.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal final class _BufferedChannel<T>: _Channel<T> {
    
    private typealias ReceiveCallback = (Result<T, CoChannelError>) -> Void
    private struct SendBlock { let element: T, resumeBlock: ((CoChannelError?) -> Void)? }
    
    private let capacity: Int
    private var receiveCallbacks = FifoQueue<ReceiveCallback>()
    private var sendBlocks = FifoQueue<SendBlock>()
    private var atomic = AtomicTuple()
    
    internal init(capacity: Int) {
        self.capacity = max(0, capacity)
    }
    
    internal override var bufferType: CoChannel<T>.BufferType {
        switch capacity {
        case .max: return .unlimited
        case 0: return .none
        case let capacity: return .buffered(capacity: capacity)
        }
    }
    
    // MARK: - send
    
    internal override func awaitSend(_ element: T) throws {
        switch atomic.update ({ count, state in
            if state != 0 { return (count, state) }
            return (count + 1, 0)
        }).old {
        case (_, 1):
            throw CoChannelError.closed
        case (_, 2):
            throw CoChannelError.canceled
        case (let count, _) where count < 0:
            receiveCallbacks.blockingPop()(.success(element))
        case (let count, _) where count < capacity:
            sendBlocks.push(.init(element: element, resumeBlock: nil))
        default:
            try Coroutine.await {
                sendBlocks.push(.init(element: element, resumeBlock: $0))
            }.map { throw $0 }
        }
    }
    
    internal override func sendFuture(_ future: CoFuture<T>) {
        future.whenSuccess { [weak self] in
            guard let self = self else { return }
            let (count, state) = self.atomic.update { count, state in
                if state != 0 { return (count, state) }
                return (count + 1, 0)
            }.old
            guard state == 0 else { return }
            count < 0
                ? self.receiveCallbacks.blockingPop()(.success($0))
                : self.sendBlocks.push(.init(element: $0, resumeBlock: nil))
        }
    }
    
    internal override func offer(_ element: T) -> Bool {
        let (count, state) = atomic.update { count, state in
            if state != 0 || count >= capacity { return (count, state) }
            return (count + 1, 0)
        }.old
        if state != 0 { return false }
        if count < 0 {
            receiveCallbacks.blockingPop()(.success(element))
            return true
        } else if count < capacity {
            sendBlocks.push(.init(element: element, resumeBlock: nil))
            return true
        }
        return false
    }
    
    // MARK: - receive
    
    internal override func awaitReceive() throws -> T {
        switch atomic.update({ count, state in
            if state == 0 { return (count - 1, 0) }
            return (Swift.max(0, count - 1), state)
        }).old {
        case (let count, let state) where count > 0:
            defer { if count == 1, state == 1 { finish() } }
            return getValue()
        case (_, 0):
            return try Coroutine.await { receiveCallbacks.push($0) }.get()
        case (_, 1):
            throw CoChannelError.closed
        default:
            throw CoChannelError.canceled
        }
    }
    
    internal override func poll() -> T? {
        let (count, state) = atomic.update { count, state in
            (Swift.max(0, count - 1), state)
        }.old
        guard count > 0 else { return nil }
        defer { if count == 1, state == 1 { finish() } }
        return getValue()
    }
    
    internal override func whenReceive(_ callback: @escaping (Result<T, CoChannelError>) -> Void) {
        switch atomic.update({ count, state in
            if state == 0 { return (count - 1, 0) }
            return (Swift.max(0, count - 1), state)
        }).old {
        case (let count, let state) where count > 0:
            callback(.success(getValue()))
            if count == 1, state == 1 { finish() }
        case (_, 0):
            receiveCallbacks.push(callback)
        case (_, 1):
            callback(.failure(.closed))
        default:
            callback(.failure(.canceled))
        }
    }
    
    internal override var count: Int {
        Int(max(0, atomic.value.0))
    }
    
    internal override var isEmpty: Bool {
        atomic.value.0 <= 0
    }
    
    private func getValue() -> T {
        let block = sendBlocks.blockingPop()
        block.resumeBlock?(nil)
        return block.element
    }
    
    // MARK: - close
    
    internal override func close() -> Bool {
        let (count, state) = atomic.update { count, state in
            state == 0 ? (Swift.max(0, count), 1) : (count, state)
        }.old
        guard state == 0 else { return false }
        if count < 0 {
            for _ in 0..<count.magnitude {
                receiveCallbacks.blockingPop()(.failure(.closed))
            }
        } else if count > 0 {
            sendBlocks.forEach { $0.resumeBlock?(.closed) }
        } else {
            finish()
        }
        return true
    }
    
    internal override var isClosed: Bool {
        atomic.value.1 == 1
    }
    
    // MARK: - cancel
    
    internal override func cancel() {
        let count = atomic.update { _ in (0, 2) }.old.0
        if count < 0 {
            for _ in 0..<count.magnitude {
                receiveCallbacks.blockingPop()(.failure(.canceled))
            }
        } else if count > 0 {
            for _ in 0..<count {
                sendBlocks.blockingPop().resumeBlock?(.canceled)
            }
        }
        finish()
    }
    
    internal override var isCanceled: Bool {
        atomic.value.1 == 2
    }
    
    deinit {
        while let block = receiveCallbacks.pop() {
            block(.failure(.canceled))
        }
        receiveCallbacks.free()
        sendBlocks.free()
    }
    
}
