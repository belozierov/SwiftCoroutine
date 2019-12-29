//
//  CoroutineContext+Pool.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 19.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

extension CoroutineContext {
    
    fileprivate static let pool = CoroutineContextPool(stackSize: .poolSize)
    
}

extension Coroutine {
    
    public static func newFromPool(dispatcher: Dispatcher,
                                   stackSize: StackSize = .recommended) -> Coroutine {
        let context = CoroutineContext.pool.pop(stackSize: stackSize)
        let coroutine = Coroutine(context: context, dispatcher: dispatcher)
        coroutine.addHandler { if $0 { CoroutineContext.pool.push(context) } }
        return coroutine
    }
    
}

public func subroutine(stackSize: Coroutine.StackSize = .recommended, block: () -> Void) {
    let context = CoroutineContext.pool.pop(stackSize: stackSize)
    CoSubroutine(context: context).start(block)
    CoroutineContext.pool.push(context)
}

extension Coroutine.StackSize {
    
    fileprivate static var poolSize: Coroutine.StackSize {
        #if os(OSX)
            return .pages(2048)
        #else
            return .pages(512)
        #endif
    }
    
}
