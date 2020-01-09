//
//  CoTransformFuture.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

final class CoTransformFuture<Input, Output>: CoFuture<Output> {
    
    typealias Transformer = (Result<Input, Error>) throws -> Output
    private let unsubscriber: (AnyHashable) -> Void
    
    @inlinable init(parent: CoFuture<Input>, transformer: @escaping Transformer) {
        unsubscriber = { [weak parent] in parent?.unsubscribe($0) }
        super.init(mutex: parent.mutex)
        parent.subscribe(with: identifier) { result in
            self.complete(with: Result { try transformer(result) })
        }
    }
    
    override func cancel() {
        unsubscriber(identifier)
        super.cancel()
    }
    
}
