//
//  CoGenerator.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 21.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

class CoGenerator<Input, Output> {
    
    func next(_ input: Input) -> Output? {
        fatalError()
    }
    
    func close() {
        fatalError()
    }
    
}

extension CoGenerator: Sequence where Input == Void {
    
    typealias Element = Output
    typealias Iterator = AnyIterator<Output>
    
    func makeIterator() -> AnyIterator<Output> {
        fatalError()
    }
    
    func next() -> Output? {
        fatalError()
    }
    
}
