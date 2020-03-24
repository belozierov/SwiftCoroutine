//
//  CoroutineProtocol.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 07.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

@usableFromInline protocol CoroutineProtocol: class {
    
    typealias StackSize = Coroutine.StackSize
    
    func await<T>(_ callback: (@escaping (T) -> Void) -> Void) -> T
    
}

extension CoroutineProtocol {
    
    @inlinable func performAsCurrent<T>(_ block: () -> T) -> T {
        let wrapper = ThreadCoroutineWrapper.current
        let caller = wrapper.coroutine
        wrapper.coroutine = self
        defer { wrapper.coroutine = caller }
        return block()
    }
    
}
