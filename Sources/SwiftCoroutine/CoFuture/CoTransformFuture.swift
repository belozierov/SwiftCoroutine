//
//  CoTransformFuture.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

final class CoTransformFuture<Input, Output>: CoFuture<Output> {
    
    typealias Transformer = (Result<Input, Error>) throws -> Output
    private weak var parent: CoFuture<Input>?
    
    @inlinable init(parent: CoFuture<Input>, transformer: @escaping Transformer) {
        self.parent = parent
        super.init(mutex: parent.mutex)
        parent.subscribe(with: self) { [unowned self] result in
            self.complete(with: Result { try transformer(result) })
        }
    }
    
    @inlinable override func cancel() {
        parent?.unsubscribe(self)
        super.cancel()
    }
    
    @inlinable public override func cancelUpstream() {
        parent?.cancelUpstream() ?? cancel()
    }
    
}
