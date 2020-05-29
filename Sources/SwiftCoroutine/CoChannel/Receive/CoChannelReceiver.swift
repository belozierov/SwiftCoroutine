//
//  CoChannelReceiver.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 07.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal final class CoChannelReceiver<T>: CoChannel<T>.Receiver {
    
    private let channel: CoChannel<T>
    
    init(channel: CoChannel<T>) {
        self.channel = channel
    }
    
    internal override var maxBufferSize: Int {
        channel.maxBufferSize
    }
    
    internal override func awaitReceive() throws -> T {
        try channel.awaitReceive()
    }
    
    internal override func receiveFuture() -> CoFuture<T> {
        channel.receiveFuture()
    }
    
    internal override func poll() -> T? {
        channel.poll()
    }
    
    internal override func whenReceive(_ callback: @escaping (Result<T, CoChannelError>) -> Void) {
        channel.whenReceive(callback)
    }
    
    internal override var count: Int {
        channel.count
    }
    
    internal override var isEmpty: Bool {
        channel.isEmpty
    }
    
    internal override var isClosed: Bool {
        channel.isClosed
    }
    
    internal override func cancel() {
        channel.cancel()
    }
    
    internal override var isCanceled: Bool {
        channel.isCanceled
    }
    
    internal override func whenComplete(_ callback: @escaping () -> Void) {
        channel.whenComplete(callback)
    }
    
    internal override func whenCanceled(_ callback: @escaping () -> Void) {
        channel.whenCanceled(callback)
    }
    
}
