//
//  CoHandleFuture.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 06.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

final class CoHandleFuture<Output>: CoFuture<Output> {
    
    @inlinable init(parent: CoFuture<Output>, handler: @escaping OutputHandler) {
        super.init(mutex: parent.mutex,
                   resultStorage: parent.$resultStorage,
                   subscriptions: parent.$subscriptions.weak)
        subscribe(with: identifier, handler: handler)
    }
    
    @inlinable override func cancel() {
        unsubscribe(identifier)?(.failure(FutureError.cancelled))
    }
    
}
