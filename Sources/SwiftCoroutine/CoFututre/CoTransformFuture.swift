//
//  CoTransformFuture.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 21.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

class CoTransformFuture<Input, Output>: CoFuture<Output> {
    
    typealias InputResult = Result<Input, Error>
    typealias Transformer = (InputResult) throws -> Output
    
    private var parent: CoFuture<Input>
    private let transformer: Transformer
    
    init(parent: CoFuture<Input>, transformer: @escaping Transformer) {
        self.parent = parent
        self.transformer = transformer
        super.init()
        addToParent()
    }
    
    // MARK: - Result
    
    @inlinable override var result: OutputResult? {
        parent.result.map { input in Result { try transformer(input) } }
    }
    
    deinit { removeFromParent() }
    
}

extension CoTransformFuture {
    
    // MARK: - Send input
    
    @inlinable func send(result: InputResult) {
        send(result: Result { try transformer(result) })
    }
    
    // MARK: - Parent completion
    
    private func addToParent() {
        parent.completions[identifier] = { [unowned self] result in
            self.send(result: Result { try self.transformer(result) })
        }
    }
    
    private func removeFromParent() {
        parent.completions[identifier] = nil
    }
    
    private var identifier: Int {
        unsafeBitCast(self, to: Int.self)
    }
    
}
