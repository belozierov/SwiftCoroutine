//
//  CoTransformFuture.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 21.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

class CoTransformFuture<Input, Output>: CoFuture<Output> {
    
    typealias InputResult = Result<Input, Error>
    typealias Transform = (InputResult) throws -> Output
    
    private var parent: CoFuture<Input>
    private let transform: Transform
    
    init(parent: CoFuture<Input>, transform: @escaping Transform) {
        self.parent = parent
        self.transform = transform
        super.init()
        addToParent()
    }
    
    // MARK: - Result
    
    @inlinable override var result: OutputResult? {
        parent.result.map { input in Result { try transform(input) } }
    }
    
    // MARK: - Transform
    
    override func transform<T>(_ transformer: @escaping (OutputResult) throws -> T) -> CoFuture<T> {
        if !isKnownUniquelyReferenced(&parent) { return super.transform(transformer) }
        let selfTransformer = self.transform
        let transformedCompletions = completions.mapValues { completion in
            { result in completion(Result { try selfTransformer(result) }) }
        }
        parent.completions.merge(transformedCompletions) { _, new in new }
        return parent.transform { result in
            try transformer(.init { try selfTransformer(result) })
        }
    }
    
    deinit { removeFromParent() }
    
}

extension CoTransformFuture {
    
    // MARK: - Send input
    
    @inlinable func send(result: InputResult) {
        send(result: Result { try transform(result) })
    }
    
    // MARK: - Parent completion
    
    private func addToParent() {
        parent.completions[identifier] = { [unowned self] result in
            self.send(result: .init { try self.transform(result) })
        }
    }
    
    private func removeFromParent() {
        parent.completions[identifier] = nil
    }
    
    private var identifier: Int {
        unsafeBitCast(self, to: Int.self)
    }
    
}
