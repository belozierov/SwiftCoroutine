//
//  CoSubroutine.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 27.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

public struct CoSubroutine {
    
    public typealias StackSize = Coroutine.StackSize
    let context: CoroutineContext
    
    init(context: CoroutineContext) {
        self.context = context
    }
    
    public init(stackSize: StackSize = .recommended) {
        self.init(context: .init(stackSize: stackSize.size))
    }
    
    public func start(_ block: () -> Void) {
        let coroutine = try? Coroutine.current()
        coroutine?.subRoutines.append(self)
        _ = withoutActuallyEscaping(block, do: context.start)
        coroutine?.subRoutines.removeLast()
    }
    
}

@inlinable public func coSubroutine<T>(_ block: () -> T) -> T {
    var result: T!
    coSubroutine { result = block() }
    return result
}

@inlinable public func coSubroutine<T>(_ block: () throws -> T) throws -> T {
    var result: Result<T, Error>!
    coSubroutine { result = Result { try block() } }
    return try result.get()
}

@inlinable public func coSubroutine<T>(_ block: @autoclosure () throws -> T) rethrows -> T {
    try coSubroutine(block())
}

