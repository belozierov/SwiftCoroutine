//
//  CoChannelReceiverWrapper.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 07.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal class CoChannelReceiverWrapper<Element>: CoChannelReceiver {
    
    func awaitReceive() throws -> Element {
        throw CoChannelError.canceled
    }
    
    func receiveFuture() -> CoFuture<Element> {
        CoFuture(_result: .failure(CoFutureError.canceled))
    }
    
    func poll() -> Element? {
        nil
    }
    
    func whenReceive(_ callback: @escaping (Result<Element, CoChannelError>) -> Void) {
        callback(.failure(.canceled))
    }
    
    func cancel() {}
    
    func whenCanceled(_ callback: @escaping () -> Void) {
        callback()
    }
    
    var count: Int { 0 }
    var isEmpty: Bool { true }
    var isClosed: Bool { true }
    var isCanceled: Bool { true }
    var maxBufferSize: Int { 0 }
    
}
