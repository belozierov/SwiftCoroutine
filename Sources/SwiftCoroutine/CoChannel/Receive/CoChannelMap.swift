//
//  CoChannelMap.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 07.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal final
class CoChannelMap<Receiver: CoChannelReceiver, Element>: CoChannelReceiverWrapper<Element> {
    
    private let receiver: Receiver
    private let transform: (Receiver.Output) -> Element
    
    internal init(receiver: Receiver, transform: @escaping (Receiver.Output) -> Element) {
        self.receiver = receiver
        self.transform = transform
    }
    
    internal override var maxBufferSize: Int {
        receiver.maxBufferSize
    }
    
    internal override func awaitReceive() throws -> Element {
        try transform(receiver.awaitReceive())
    }
    
    internal override func receiveFuture() -> CoFuture<Element> {
        receiver.receiveFuture().map(transform)
    }
    
    internal override func poll() -> Element? {
        receiver.poll().map(transform)
    }
    
    internal override func whenReceive(_ callback: @escaping (Result<Element, CoChannelError>) -> Void) {
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
    
    internal override func whenCanceled(_ callback: @escaping () -> Void) {
        receiver.whenCanceled(callback)
    }
    
}
