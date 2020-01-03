//
//  CoTransformFuture2.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

class CoTransformFuture<Input, Output>: CoFuture<Output> {
    
    typealias Transformer = (Result<Input, Error>) throws -> Output
    
    private let parent: CoFuture<Input>
    private let transformer: Transformer
    
    init(parent: CoFuture<Input>, transformer: @escaping Transformer) {
        self.parent = parent
        self.transformer = transformer
        super.init(mutex: parent.mutex)
        addToParent()
    }
    
    override var result: OutputResult? {
        parent.result.map { input in Result { try transformer(input) } }
    }
    
    deinit { removeFromParent() }
    
}

extension CoTransformFuture {
    
    // MARK: - Parent completion

    private func addToParent() {
        parent.subscribe(with: identifier) { [unowned self] result in
            self.complete(with: Result { try self.transformer(result) })
        }
    }

    private func removeFromParent() {
        parent.unsubscribe(identifier)
    }

    private var identifier: Int {
        unsafeBitCast(self, to: Int.self)
    }

}
