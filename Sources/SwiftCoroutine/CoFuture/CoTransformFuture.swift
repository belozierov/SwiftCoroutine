//
//  CoTransformFuture.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

class CoTransformFuture<Input, Output>: CoFuture<Output> {
    
    typealias Transformer = (Result<Input, Error>) throws -> Output
    
    private let parent: CoFuture<Input>
    private let transformer: Transformer
    
    @inlinable init(parent: CoFuture<Input>, transformer: @escaping Transformer) {
        self.parent = parent
        self.transformer = transformer
        super.init(mutex: parent.mutex)
        subscribe()
    }
    
    @inlinable override var result: OutputResult? {
        parent.result.map { input in Result { try transformer(input) } }
    }
    
    private func subscribe() {
        parent.subscribe(with: identifier) { [unowned self] result in
            self.complete(with: Result { try self.transformer(result) })
        }
    }
    
    deinit {
        parent.unsubscribe(identifier)
    }
    
}
