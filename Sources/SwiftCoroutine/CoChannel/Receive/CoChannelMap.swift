//
//  CoChannelMap.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 07.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal final class CoChannelMap<R: CoChannelReceiverProtocol, T>: CoChannel<T>.Receiver {
    
    private let receiver: R
    private let transform: (R.Output) -> T
    
    internal init(receiver: R, transform: @escaping (R.Output) -> T) {
        self.receiver = receiver
        self.transform = transform
    }
    
    internal override var maxBufferSize: Int {
        receiver.maxBufferSize
    }
    
    internal override func awaitReceive() throws -> T {
        try transform(receiver.awaitReceive())
    }
    
    internal override func receiveFuture() -> CoFuture<T> {
        receiver.receiveFuture().map(transform)
    }
    
    internal override func poll() -> T? {
        receiver.poll().map(transform)
    }
    
    internal override func whenReceive(_ callback: @escaping (Result<T, CoChannelError>) -> Void) {
        receiver.whenReceive { callback($0.map(self.transform)) }
    }
    
    internal override var count: Int {
        receiver.count
    }
    
    internal override var isEmpty: Bool {
        receiver.isEmpty
    }
    
    internal override var isClosed: Bool {
        receiver.isClosed
    }
    
    internal override func cancel() {
        receiver.cancel()
    }
    
    internal override var isCanceled: Bool {
        receiver.isCanceled
    }
    
    internal override func whenComplete(_ callback: @escaping () -> Void) {
        receiver.whenComplete(callback)
    }
    
}
