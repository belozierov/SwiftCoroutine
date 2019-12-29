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

public func subroutine(stackSize: Coroutine.StackSize = .recommended, block: () -> Void) {
    CoSubroutine(stackSize: stackSize).start(block)
}

@inlinable public func subroutine<T>(stackSize: Coroutine.StackSize = .recommended,
                                     block: () -> T) -> T {
    var result: T!
    subroutine(stackSize: stackSize) { result = block() }
    return result
}

@inlinable public func subroutine<T>(stackSize: Coroutine.StackSize = .recommended,
                                     block: () throws -> T) throws -> T {
    var result: Result<T, Error>!
    subroutine(stackSize: stackSize) { result = Result { try block() } }
    return try result.get()
}

@inlinable public func subroutine<T>(stackSize: Coroutine.StackSize = .recommended,
                                       block: @autoclosure () throws -> T) rethrows -> T {
    try subroutine(stackSize: stackSize, block: block())
}

