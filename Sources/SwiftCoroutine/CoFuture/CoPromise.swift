//
//  CoPromise.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

final public class CoPromise<Output>: CoFuture<Output> {
    
    public init() {
        super.init()
    }
    
}

extension CoPromise: CoSubject {
    
    @inlinable public func send(completion: OutputResult) {
        complete(with: completion)
    }
    
}
