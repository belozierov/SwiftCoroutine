//
//  CoTransformFuture.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 20.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

class CoTransformFuture<Input, Output>: CoFuture<Output> {
    
    typealias InputResult = Swift.Result<Input, Error>
    typealias Transformer = (InputResult) throws -> Output
    
    private let transformer: Transformer
    private var parent: CoFuture<Input>
    
    init(parent: CoFuture<Input>, transformer: @escaping Transformer) {
        self.transformer = transformer
        self.parent = parent
        super.init()
        parent.setSubscription(for: identifier) { [unowned self] result in
            self.finish(with: .init { try transformer(result) })
        }
    }
    
    override func cancel() {
        super.cancel()
        parent.cancel()
    }
    
    override func transform<T>(_ transformer: @escaping (Result) throws -> T) -> CoFuture<T> {
        if !isKnownUniquelyReferenced(&parent) { return super.transform(transformer) }
        let selfTransformer = self.transformer
        return parent.transform { result in
            try transformer(.init { try selfTransformer(result) })
        }
    }
    
    private var identifier: Int {
        unsafeBitCast(self, to: Int.self)
    }
    
    deinit {
        parent.setSubscription(for: identifier, completion: nil)
    }
    
}
