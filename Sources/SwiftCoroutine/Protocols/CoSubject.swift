//
//  CoSubject.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

public protocol CoSubject: CoPublisher {
    
    func send(_ value: Output)
    func send(completion: OutputResult)
    
}

extension CoSubject {
    
    @inlinable public func send(_ value: Output) {
        send(completion: .success(value))
    }
    
    @inlinable public func send(_ error: Error) {
        send(completion: .failure(error))
    }
    
    @inlinable public func perform(_ block: () throws -> Output) {
        send(completion: Result(catching: block))
    }
    
}
