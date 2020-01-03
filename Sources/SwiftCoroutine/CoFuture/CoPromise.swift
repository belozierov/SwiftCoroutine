//
//  CoPromise.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

public class CoPromise<Output>: CoFuture<Output> {
    
    private var _result: OutputResult?
    
    @inlinable public init() {
        super.init()
    }
    
    public override var result: Result<Output, Error>? {
        mutex.lock()
        defer { mutex.unlock() }
        return _result
    }
    
    override func saveResult(_ result: OutputResult) {
        _result = result
    }
    
}

extension CoPromise: CoSubject {
    
    @inlinable public func send(completion: OutputResult) {
        complete(with: completion)
    }
    
}
