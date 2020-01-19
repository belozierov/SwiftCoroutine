//
//  CoHandleFuture.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 06.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

final class CoHandleFuture<Output>: CoFuture<Output> {
    
    private let subscribeIdentifier: AnyHashable
    private weak var parent: CoFuture<Output>?
    
    @inlinable init(parent: CoFuture<Output>, handler: @escaping OutputHandler) {
        subscribeIdentifier = parent.addHandler(handler)
        self.parent = parent
        super.init(mutex: parent.mutex,
                   resultStorage: parent.$resultStorage,
                   subscriptions: parent.$subscriptions.weak)
    }
    
    @inlinable override func cancel() {
        mutex.lock()
        guard resultStorage == nil else { return mutex.unlock() }
        newResultStorage(with: .failure(CoFutureError.cancelled))
        let handler = subscriptions?.removeValue(forKey: subscribeIdentifier)
        mutex.unlock()
        handler?(.failure(CoFutureError.cancelled))
    }
    
    @inlinable override func cancelUpstream() {
        parent?.cancelUpstream() ?? cancel()
    }
    
}
